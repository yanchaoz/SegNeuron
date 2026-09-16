"""Raw-only SegNeuron inference, retaining the published MNet and patch settings."""

import argparse
from collections.abc import Mapping
import hashlib
from itertools import product
import json
from pathlib import Path
import time


CROP_SIZE = (20, 128, 128)
STRIDE = (10, 64, 64)


def load_volume(path):
    """Read a grayscale uint8 ZYX volume without silently scaling its intensities."""
    import numpy as np

    path = Path(path)
    if path.suffix.lower() == ".npy":
        volume = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(path) as image:
            if len(image.series) != 1 or any(axis in image.series[0].axes for axis in "CST"):
                raise ValueError("TIFF must contain one grayscale ZYX stack, without color, channel or time axes")
            volume = image.series[0].asarray()
    else:
        raise ValueError("Input must be a .npy, .tif or .tiff volume")
    validate_volume(volume)
    return volume


def validate_volume(volume):
    import numpy as np

    if volume.ndim != 3 or any(size == 0 for size in volume.shape):
        raise ValueError("Input must be a non-empty 3D grayscale volume in Z,Y,X order")
    if volume.dtype != np.uint8:
        raise ValueError("Input must have uint8 dtype; convert intensities explicitly before inference")


def tile_layout(shape):
    """Use the original overlapping grid, including safe padding for tiny/odd inputs."""
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("Expected three positive Z,Y,X dimensions")
    counts = tuple(max(1, (n - c) // s + 2) for n, c, s in zip(shape, CROP_SIZE, STRIDE))
    padded_shape = tuple(c + (count - 1) * s for c, count, s in zip(CROP_SIZE, counts, STRIDE))
    padding = tuple(((p - n) // 2, (p - n + 1) // 2) for n, p in zip(shape, padded_shape))
    starts = tuple(tuple(i * step for i in range(count)) for count, step in zip(counts, STRIDE))
    return padding, starts


def gaussian_weight():
    """Original Gaussian blending weights (sigma=0.2, nonzero floor=1e-6)."""
    import numpy as np

    zz, yy, xx = np.meshgrid(
        *(np.linspace(-1, 1, n, dtype=np.float32) for n in CROP_SIZE), indexing="ij"
    )
    distance = np.sqrt(zz * zz + yy * yy + xx * xx)
    return 1e-6 + np.exp(-(distance ** 2 / (2.0 * 0.2 ** 2)))


def infer_volume(model, volume, device="cpu", progress=None):
    """Return float32 affinities (3,Z,Y,X) and boundaries (Z,Y,X), without GT."""
    import numpy as np
    import torch

    validate_volume(volume)
    padding, starts = tile_layout(volume.shape)
    padded = np.pad(volume, padding, mode="reflect")
    sums = np.zeros((4,) + padded.shape, dtype=np.float32)
    weights = np.zeros(padded.shape, dtype=np.float32)
    patch_weight = gaussian_weight()
    total = len(starts[0]) * len(starts[1]) * len(starts[2])
    model = model.to(device).eval()

    with torch.inference_mode():
        for index, position in enumerate(product(*starts), 1):
            region = tuple(slice(start, start + size) for start, size in zip(position, CROP_SIZE))
            patch = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
            tensor = torch.from_numpy(patch[None, None]).to(device)
            affinities, boundaries = model(tensor)
            expected = (1, 3) + CROP_SIZE, (1, 1) + CROP_SIZE
            if tuple(affinities.shape) != expected[0] or tuple(boundaries.shape) != expected[1]:
                raise ValueError("Model output shapes must be (1,3,20,128,128) and (1,1,20,128,128)")
            prediction = torch.cat((affinities, boundaries), dim=1)[0].float().cpu().numpy()
            if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
                raise ValueError("Model output must contain finite probabilities in [0,1]")
            sums[(slice(None),) + region] += prediction * patch_weight
            weights[region] += patch_weight
            if progress is not None:
                progress(index, total)

    if not np.all(weights > 0):
        raise RuntimeError("Inference grid did not cover the volume")
    sums /= weights[None]
    # A convex blend is a probability; remove only float32 accumulation roundoff.
    np.clip(sums, 0.0, 1.0, out=sums)
    # An explicit end preserves singleton axes and avoids the empty [0:-0] case.
    original = tuple(slice(pad[0], pad[0] + size) for pad, size in zip(padding, volume.shape))
    affinities = sums[(slice(0, 3),) + original].copy()
    boundaries = sums[(3,) + original].copy()
    return affinities, boundaries


def load_checkpoint(model, path):
    """Strictly load official model_weights, state_dict, or a bare tensor state dict."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state dictionary or contain model_weights/state_dict")
    state = checkpoint.get("model_weights", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint has no non-empty model state dictionary")
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("Model state must map parameter names to tensors")
        name = key[7:] if key.startswith("module.") else key
        if name in cleaned:
            raise ValueError("Checkpoint has duplicate parameter names after removing module. prefix")
        cleaned[name] = value
    model.load_state_dict(cleaned, strict=True)


def select_device(name):
    import torch

    device = torch.device(name)
    if device.type == "cpu" and device.index is None:
        return device
    if device.type != "cuda" or device.index is None:
        raise ValueError("Device must be cpu or cuda:N (for example cuda:0)")
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise ValueError(f"Requested device {name} is not available")
    return device


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="uint8 3D ZYX TIFF or NPY raw volume")
    parser.add_argument("--checkpoint", required=True, type=Path, help="MNet checkpoint (.pth or .pt)")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing paths are refused")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:N, e.g. cuda:0")
    args = parser.parse_args(argv)
    for path, label in ((args.input, "Input"), (args.checkpoint, "Checkpoint")):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}; choose a new directory")

    # Keep --help and argument/path checks usable before installing model dependencies.
    import numpy as np
    import tifffile
    import torch
    if __package__:
        from .model.Mnet import MNet
    else:
        from model.Mnet import MNet

    try:
        device = select_device(args.device)
        raw = load_volume(args.input)
        model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
        load_checkpoint(model, args.checkpoint)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))

    started = time.perf_counter()
    print(f"Input {raw.shape} uint8 ZYX; device={device}", flush=True)
    affinities, boundaries = infer_volume(
        model, raw, device, progress=lambda i, n: print(f"Patch {i}/{n}", flush=True)
    )
    elapsed = time.perf_counter() - started
    # Exclusive creation also catches a concurrent run choosing the same destination.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "affinities.npy", affinities, allow_pickle=False)
    tifffile.imwrite(args.output_dir / "boundaries.tif", boundaries, photometric="minisblack", metadata={"axes": "ZYX"})
    padding, starts = tile_layout(raw.shape)
    manifest = {
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input), "shape_zyx": list(raw.shape), "dtype": "uint8"},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "model": "MNet(1, kn=(32,64,96,128,256), FMU='sub')",
        "device": str(device), "torch_version": str(torch.__version__),
        "crop_size_zyx": CROP_SIZE, "stride_zyx": STRIDE, "padding_zyx": padding,
        "normalization": "float32(raw) / 255.0", "blend": "Gaussian sigma=0.2, floor=1e-6",
        "patch_count": len(starts[0]) * len(starts[1]) * len(starts[2]), "inference_seconds": elapsed,
        "affinities": {"file": "affinities.npy", "axes": "CZYX", "offsets_zyx": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], "dtype": "float32"},
        "boundaries": {"file": "boundaries.tif", "axes": "ZYX", "dtype": "float32"},
    }
    # Written last: this file records completion of both output writes.
    with (args.output_dir / "inference.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Saved affinities.npy, boundaries.tif and inference.json to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
