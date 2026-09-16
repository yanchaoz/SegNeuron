"""ELF watershed + multicut postprocessing for SegNeuron probability outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def _probabilities(values, name, ndim):
    values = np.asarray(values)
    if values.ndim != ndim or any(size == 0 for size in values.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-D array")
    if values.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain real probabilities")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return np.ascontiguousarray(values, dtype=np.float32)


def _load_elf():
    # Keep --help and input validation usable without the compiled ELF backend.
    try:
        import elf.segmentation.features as feats
        import elf.segmentation.multicut as mc
        import elf.segmentation.watershed as ws
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ELF multicut is unavailable. Install a compatible python-elf, "
            "nifty and vigra environment (see README); no fallback is used. "
            f"Backend error: {exc}"
        ) from exc
    return feats, mc, ws


def post_mc(affs, beta=0.25):
    """Segment merge affinities of shape (3, z, y, x), returning uint32 IDs.

    Channels correspond to offsets (-1,0,0), (0,-1,0), (0,0,-1).
    The original watershed settings and ELF Kernighan-Lin multicut are kept.
    Output IDs start at 1; cluster 0 from the solver is a neuron, not background.
    """
    affs = _probabilities(affs, "affinities", 4)
    if affs.shape[0] != 3:
        raise ValueError("affinities must have shape (3, z, y, x)")
    if isinstance(beta, (bool, np.bool_)) or not np.isscalar(beta):
        raise ValueError("beta must be a finite number strictly between 0 and 1")
    try:
        beta = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be a finite number strictly between 0 and 1") from exc
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta must be a finite number strictly between 0 and 1")

    feats, mc, ws = _load_elf()
    split_affs = 1.0 - affs
    boundary_input = np.maximum(split_affs[1], split_affs[2])
    watershed = np.empty(boundary_input.shape, dtype=np.uint64)
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, _ = ws.distance_transform_watershed(
            boundary_input[z], threshold=0.25, sigma_seeds=2.0
        )
        wsz = np.asarray(wsz)
        if wsz.shape != watershed.shape[1:] or wsz.dtype.kind not in "ui" or np.any(wsz <= 0):
            raise RuntimeError(f"ELF watershed returned invalid or unassigned fragments at z={z}")
        # Dense zero-based RAG nodes, disjoint across slices even for sparse IDs.
        ids, inverse = np.unique(wsz, return_inverse=True)
        watershed[z] = inverse.reshape(wsz.shape).astype(np.uint64) + offset
        offset += len(ids)

    rag = feats.compute_rag(watershed)
    if rag.numberOfEdges == 0:
        # Exact edgeless-graph solution; ELF's cost scaling needs nonempty edges.
        node_labels = np.arange(rag.numberOfNodes, dtype=np.uint64)
    else:
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        costs = feats.compute_affinity_features(rag, split_affs, offsets)[:, 0]
        edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
        if not np.isfinite(costs).all() or not np.isfinite(edge_sizes).all() or np.any(edge_sizes <= 0):
            raise RuntimeError("ELF returned invalid edge probabilities or sizes")
        costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)
        node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    labels, inverse = np.unique(segmentation, return_inverse=True)
    if len(labels) > np.iinfo(np.uint32).max:
        raise OverflowError("Too many instances for uint32 output")
    return (inverse.reshape(watershed.shape) + 1).astype(np.uint32)


def _load_volume(path):
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    raise ValueError(f"Expected .npy, .tif or .tiff: {path}")


def _validate_ground_truth(gt, shape):
    if gt.shape != shape:
        raise ValueError(f"ground truth shape {gt.shape} does not match {shape}")
    if gt.dtype.kind not in "ui" or np.any(gt < 0):
        raise ValueError("ground truth must contain nonnegative integer instance IDs")
    if not np.any(gt):
        raise ValueError("ground truth contains only ignored label 0")
    # Metric implementations index by label: compact sparse IDs without dropping background.
    ids, inverse = np.unique(gt, return_inverse=True)
    return (inverse.reshape(gt.shape) + int(ids[0] != 0)).astype(np.uint64)


def _write_labels(path, segmentation):
    # Exclusive creation also protects against a file appearing during computation.
    with path.open("xb") as stream:
        try:
            if path.suffix.lower() == ".npy":
                np.save(stream, segmentation, allow_pickle=False)
            else:
                import tifffile
                tifffile.imwrite(stream, segmentation, photometric="minisblack", metadata={"axes": "ZYX"})
        except BaseException:
            stream.close()
            path.unlink()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinities", type=Path, required=True, help="NPY merge probabilities, shape (3,z,y,x)")
    parser.add_argument("--boundaries", type=Path, required=True,
                        help="TIFF foreground-head probabilities (original boundary output), shape (z,y,x)")
    parser.add_argument("--output", type=Path, required=True, help="New uint32 labels file (.tif, .tiff or .npy)")
    parser.add_argument("--beta", type=float, default=0.25, help="Multicut bias strictly between 0 and 1 (default: 0.25)")
    parser.add_argument("--ground-truth", type=Path, help="Optional same-grid neuron instance labels; 0 is ignored")
    args = parser.parse_args(argv)

    try:
        if args.output.suffix.lower() not in (".npy", ".tif", ".tiff"):
            raise ValueError("output must use .npy, .tif or .tiff")
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        if not args.output.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {args.output.parent}")
        if args.affinities.suffix.lower() != ".npy":
            raise ValueError("affinities must be a .npy file")
        if args.boundaries.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("boundaries must be a .tif or .tiff file")
        affinities = _probabilities(_load_volume(args.affinities), "affinities", 4)
        boundaries = _probabilities(_load_volume(args.boundaries), "boundaries", 3)
        if affinities.shape[0] != 3 or affinities.shape[1:] != boundaries.shape:
            raise ValueError("affinities must be (3,z,y,x) and boundaries must match (z,y,x)")
        gt = None
        if args.ground_truth is not None:
            gt = _validate_ground_truth(_load_volume(args.ground_truth), boundaries.shape)
            from skimage.metrics import adapted_rand_error, variation_of_information

        # Keep the original foreground-head fusion and probability direction.
        segmentation = post_mc(np.minimum(affinities, boundaries[None]), args.beta)
        result = {"output": str(args.output), "shape": list(segmentation.shape),
                  "dtype": str(segmentation.dtype), "instances": int(segmentation.max()), "beta": args.beta}
        if gt is not None:
            arand = adapted_rand_error(gt, segmentation, ignore_labels=(0,))[0]
            voi_split, voi_merge = variation_of_information(gt, segmentation, ignore_labels=(0,))
            metrics = {"arand": float(arand), "voi_split": float(voi_split),
                       "voi_merge": float(voi_merge), "voi": float(voi_split + voi_merge)}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("Ground-truth metrics are undefined for this volume")
            result["metrics"] = metrics
        _write_labels(args.output, segmentation)
    except (ValueError, OSError, RuntimeError, ImportError, OverflowError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
