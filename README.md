
# SegNeuron  <img src="/Figures/logo.png" alt="logo" width="50" style="vertical-align: middle;"/> 

> [!IMPORTANT]
> This is a legacy-stabilized copy of upstream commit `659ce323`. The original
> commit is pinned by the `legacy/upstream-659ce323` branch and documented in
> [`legacy/`](legacy/README.md). Stabilization
> fixes startup crashes and configuration errors while intentionally preserving
> the model, checkpoint, inference, loss, normalization, postprocessing, and
> input/output contracts. Use `environment.yml` for the reproducible legacy
> environment; the original full environment export remains in
> `requirements.txt` for provenance.

The unmodified upstream commit is retained as the legacy baseline at
[`legacy/upstream-659ce323`](https://github.com/yanchaoz/SegNeuron/tree/legacy/upstream-659ce323).
The exact commit and reference-model hashes are recorded in
[`legacy/manifest.json`](legacy/manifest.json) and enforced by the contract
tests. Do not rewrite the legacy branch when updating maintained code; add a
reviewed migration and golden evidence instead.

> [!NOTE]
> This code-polished edition also applies behavior-preserving formatting, import
> cleanup, safer function defaults, narrower exception handling, and fixes for
> deterministic Python errors across the repository. See
> [`CODE_QUALITY_REPORT.md`](CODE_QUALITY_REPORT.md) for scope, remaining risks,
> and verification evidence. Run `python -m ruff check .` and
> `python -B -m unittest discover -s tests -v` before making further changes.

### Verified compatibility

The maintained code was smoke-tested on the following legacy GPU environment:

| Component | Verified value |
| --- | --- |
| Python | 3.8.10 |
| PyTorch | 1.9.0+cu102 |
| CUDA | available; CUDA forward pass verified |
| Contract/model-equivalence tests | 8/8 passed |

Both supervised and pretraining `MNet` variants completed a CUDA forward pass
with input shape `(1, 1, 8, 32, 32)`. This is a compatibility smoke test, not a
substitute for end-to-end validation with the published datasets and weights.

Official implementation, datasets and trained models of "SegNeuron: 3D Neuron Instance Segmentation in
 Any EM Volume with a Generalist Model" （[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/0518_paper.pdf)) 

![GitHub stars](https://img.shields.io/github/stars/yanchaoz/SegNeuron?style=social)
![GitHub forks](https://img.shields.io/github/forks/yanchaoz/SegNeuron?style=social)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=yanchaoz.SegNeuron)
[![HuggingFace](https://img.shields.io/badge/🤗%20Dataset-EMNeuron-yellow)](https://huggingface.co/datasets/yanchaoz/EMNeuron)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yanchaoz/SegNeuron/blob/main/SegNeuron_Colab_Inference.ipynb)


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



 

## Environments
We have packaged all the dependencies into Connect.tar.gz, which can be directly downloaded for easy access [here](https://huggingface.co/yanchaoz/SegNeuron).
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



## Training
### 1. Pretraining
```
cd Pretrain
```
```
python pretrain.py
```
### 2. Supervised Training
```
cd Train_and_Inference
```
```
python supervised_train.py
```
## Inference
### 1. Affinity Inference
```
cd Train_and_Inference
```
```
python inference.py
```
### 2. Instance Segmentation
```
cd Postprocess
```
```
python FRMC_post.py
```
### 3. Zero-shot Segmentation Examples on [MitoEM](https://mitoem.grand-challenge.org/) and [Wildenberg](https://bossdb.org/project/wildenberg2023) (scale bar: 2 um)
<p align="center">
  <img src="/Figures/example.png"  alt="" width="700"/>
</p>

## Acknowledgement
This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf). Should you have any further questions, please let us know. Thanks again for your interest.
