
# SegNeuron  <img src="/Figures/logo.png" alt="logo" width="50" style="vertical-align: middle;"/> 
Official implementation, datasets and trained models of "SegNeuron: 3D Neuron Instance Segmentation in
 Any EM Volume with a Generalist Model" （[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/0518_paper.pdf)) 

![GitHub stars](https://img.shields.io/github/stars/yanchaoz/SegNeuron?style=social)
![GitHub forks](https://img.shields.io/github/forks/yanchaoz/SegNeuron?style=social)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=yanchaoz.SegNeuron)
[![HuggingFace](https://img.shields.io/badge/🤗%20Dataset-EMNeuron-yellow)](https://huggingface.co/datasets/yanchaoz/EMNeuron)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yanchaoz/SegNeuron/blob/main/SegNeuron_Colab_Inference.ipynb)

The Colab notebook is the original self-contained demonstration with its own environment. For the maintained command-line workflow, use the installation and inference instructions below.


> [!TIP]
> **SegNeuron is now available as an Agent Skill in [EM-Skills](https://github.com/yanchaoz/EM-Skills).**
>
> **EM-Skills** is a growing collection of reusable Agent Skills for EM analysis, currently including:
> - 🧠 **Neuron Segmentation Skill** — SegNeuron-based neuron reconstruction
> - 🧬 **Mitochondria Segmentation Skill** — MitoNet-based mitochondria segmentation
> - 🎯 **Annotation Selection Skill** — informative region selection for efficient annotation
> - 🎥 **EM Visualization Skill** — CloudVolume-based visualization and video generation
>
> 👉 See **[EM-Skills](https://github.com/yanchaoz/EM-Skills)** for installation and usage. Contributions are welcome—feel free to submit issues, propose new EM Skills, or improve existing ones.


## How does SegNeuron speed up neuron segmentation in EM volumes?
The general-purpose model achieves outstanding reconstruction performance on entirely unseen 3D EM datasets (x/y resolution: **5–10** nm). Human experts only need to perform connectivity corrections on the coarse segmentation results, which can then be directly used to fine-tune SegNeuron or to train new lightweight models. 
<p align="center">
  <img src="/Figures/pipeline.png"  alt="SegNeuron-based Pipeline" width="900"/>
</p>



 

## Installation

Use Python 3.10 or newer for affinity inference. From the repository root:

```bash
git clone https://github.com/yanchaoz/SegNeuron.git
cd SegNeuron
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Install the appropriate PyTorch build using the [official selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

For CPU-only inference, an explicit PyTorch installation is:

```bash
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

FRMC instance segmentation uses the original ELF/nifty/vigra backend in a separate Linux conda environment:

```bash
conda env create -f environment-postprocess.yml
conda activate segneuron-postprocess
python -c "import elf.segmentation.multicut, elf.segmentation.features, elf.segmentation.watershed"
```

The environment pins `python-elf=0.8.1`: [ELF 0.9 changed its C++ backend](https://github.com/constantinpape/elf), so upgrading it is a separate compatibility change. On Windows, run this postprocessing environment in Linux/WSL; a successful `pip install` alone does not establish that its compiled dependencies work. The scripts report missing or broken dependencies rather than substituting a different segmentation algorithm.

The original source, README, notebook and environment freeze are preserved in [`legacy/`](legacy/) from commit `ccb0ba2c5e28e0d2c454e7320c341e71f4eb148c`. The historical `Connect.tar.gz` environment remains available from the [model repository](https://huggingface.co/yanchaoz/SegNeuron); it is not required for the inference commands below.

## Datasets and Models
The datasets required for model development and validation are available [here](https://huggingface.co/datasets/yanchaoz/EMNeuron). The trained models can be download [here](https://huggingface.co/yanchaoz/SegNeuron). If you use any of the following vEM datasets in your work, please also cite the corresponding original publications:

- **vEM1: MiRA-ADWT**  
  *Ultrastructural Alterations of Dendritic Morphology in the Prefrontal Cortex of Alzheimer’s Disease Model Rats* [link](https://link.springer.com/article/10.1007/s12264-026-01606-5)

- **vEM2: MiRA-ZF**  
  *Multiplexed Neuromodulatory-Type-Annotated EM-Reconstruction of Larval Zebrafish* [link](https://www.biorxiv.org/content/10.1101/2025.06.12.659365v1)

- **vEM3: MiRA-SCN**  
  *Connectomic Organization of the Suprachiasmatic Nucleus* [link](https://www.biorxiv.org/content/10.1101/2024.10.20.619252v1)

- **vEM4: MiRA-PIB**   
  *PIB: Parallel ion beam etching of sections collected on wafer for ultra large-scale connectomics* [link](https://www.biorxiv.org/content/10.1101/2025.04.25.650569v4)

### Table: Details of EMNeuron

<div style="font-size: 0.6em;">

| Dataset              | Modality   | Res.($nm$) ($x\/y,z$) | Total voxels (M) | Labeled voxels (M) | Dataset               | Modality   | Res.($nm$) ($x\/y,z$) | Total voxels (M) | Labeled voxels (M) |
|----------------------|------------|----------------------|------------------|--------------------|-----------------------|------------|----------------------|------------------|--------------------|
| ZFinch       | SBF-SEM    | 9, 20             | 3635             | 131                | HBrain         | FIB-SEM    | 8, 8              | 3072             | 844                |
| Layer4        | SBF-SEM    | 9, 20             | 1674             | -                  | FIB25        | FIB-SEM    | 8, 8              | 312              | 312                |
| _vEM1_ (adwt)            | ATUM-SEM   | 8, 50             | 1205             | 157                |  Minnie       | ssTEM      | 8, 40             | 2096             | -                  |
| _vEM2_ (zfish)            | ATUM-SEM   | 8, 30             | 1329             | 281                |  Pinky        | ssTEM      | 8, 40             | 1165             | 117                |
| _vEM3_ (scn)            | ATUM-SEM   | 8, 40             | 1301             | 253                |  FAFB         | ssTEM      | 8, 40             | 2625             | 577                |
|MitoEM        | ATUM-SEM   | 8, 30             | 1048             | -                  |  Basil        | ssTEM      | 8, 40             | 23               | 23                 |
| H01           | ATUM-SEM   | 8, 30             | 1166             | 118                |  Harris       | others     | 6, 50             | 30               | 30                 |
| Kasthuri      | ATUM-SEM   | 6, 30             | 1526             | 478                |  _vEM4_ (ionsem)            | others     | 8, 20             | 45               | -                  |

</div>



## Inference

### 1. Affinity inference

Download [`SegNeuronModel.ckpt`](https://huggingface.co/yanchaoz/SegNeuron/resolve/main/SegNeuronModel.ckpt) to `weights/SegNeuronModel.ckpt`. Input is a nonempty **3D `uint8` TIFF or NPY volume in `(z, y, x)` order**, without a channel or time axis. Choose the imaging scale before running inference; the script does not resize, resample or reinterpret axes. The paper targets approximately 5–10 nm x/y sampling.

Run from the repository root in the inference environment:

```bash
python Train_and_Inference/inference.py --input data/raw.tif --checkpoint weights/SegNeuronModel.ckpt --output-dir runs/example --device cpu
```

For a CUDA-enabled PyTorch installation, use `--device cuda:0`. The output directory must not already exist.

Inference retains MNet's architecture, divides intensities by 255, and blends overlapping `20 × 128 × 128` tiles with Gaussian weights and stride `10 × 64 × 64`. Small inputs are padded for the model and outputs are cropped back to the original shape. Inputs and output accumulators are held in RAM; begin with a representative crop before processing a large volume.

| Output | Meaning |
|---|---|
| `affinities.npy` | `float32` probabilities, shape `(3, z, y, x)`; channels connect to offsets `(-1,0,0)`, `(0,-1,0)`, `(0,0,-1)` |
| `boundaries.tif` | `float32` auxiliary-head probabilities, shape `(z, y, x)`; historical filename retained |
| `inference.json` | Run settings and input/checkpoint hashes |

The auxiliary head is trained against `label != 0`; despite its historical filename, its output is foreground/interior confidence, not a membrane probability to invert. FRMC combines it with each affinity channel using the original elementwise minimum. Keep both files from the same run and grid.

### 2. Instance segmentation

Activate `segneuron-postprocess` and run from the repository root:

```bash
python Postprocess/FRMC_post.py --affinities runs/example/affinities.npy --boundaries runs/example/boundaries.tif --output runs/example/segmentation-beta025.tif --beta 0.25
```

The output is a `uint32` neuron-instance label volume with positive IDs on the same `(z, y, x)` grid. NPY output is also supported by choosing a `.npy` filename. Its parent directory must exist and an existing output file is refused. A JSON summary is printed to stdout. Ground truth is optional; add `--ground-truth data/neuron-labels.tif` only when it contains matching neuron-instance annotations on the same grid. This adds adapted Rand error and split/merge variation of information to the summary. Synapse or mitochondria annotations are not neuron ground truth.

To compare parameters, repeat postprocessing with a different `--beta` and output filename; reuse the same inference outputs. The default `0.25` reproduces the original parameter, not an accuracy guarantee on a new dataset. Inspect merge/split errors across slices and use neuron ground truth to compare accuracy when available.

### 3. Zero-shot Segmentation Examples on [MitoEM](https://mitoem.grand-challenge.org/) and [Wildenberg](https://bossdb.org/project/wildenberg2023) (scale bar: 2 um)
<p align="center">
  <img src="/Figures/example.png"  alt="" width="700"/>
</p>

## Training

Training retains the original research data layout and CUDA workflow. Install a CUDA-compatible PyTorch build and `python -m pip install -r requirements-training.txt` in the training environment. Before launching, edit `config/SegNeuron.yaml` inside the corresponding training directory: set `DATA.data_folder`, checkpoint paths and flags under `MODEL`, and an appropriate batch size and worker count. Set `MODEL.pre_train: False` when starting without pretrained weights. `--cfg SegNeuron` selects that YAML basename; it is not a filesystem path.

Pretraining expects numerically named volumes (`0.tif`, `1.tif`, ...) in the 13 dataset directories listed in [`pretrain_provider.py`](Pretrain/pretrain_provider.py). Supervised training expects raw/instance pairs (`0.tif`, `0_MaskIns.tif`, ...) in the 10 directories listed in [`supervised_provider.py`](Train_and_Inference/supervised_provider.py). Its dataset-balanced sampling retains the original **76-volume order**, with per-directory counts `33, 2, 6, 3, 9, 9, 3, 2, 5, 4`; arbitrary custom folder collections require adapting that sampler. Keep these folders free of unrelated files. Training crops are `20 × 128 × 128`; volumes must remain large enough after the provider's z-subsampling augmentation.

Run each stage from its own directory:

```bash
cd Pretrain
python pretrain.py --cfg SegNeuron
```

```bash
cd Train_and_Inference
python supervised_train.py --cfg SegNeuron
```

The two `cd` commands above are relative to the repository root. Set `CUDA_VISIBLE_DEVICES` before launching to choose GPUs. The retained loops write training losses, previews and checkpoints; the historical validation-related YAML fields do not implement a held-out validation loop. Full retraining and reproduction of the paper's benchmarks require the research datasets and are separate from inference smoke tests.

## Acknowledgement
This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf). Should you have any further questions, please let us know. Thanks again for your interest.
