
# SegNeuron  <img src="/Figures/logo.png" alt="logo" width="50" style="vertical-align: middle;"/> 
Official implementation, datasets and trained models of "SegNeuron: 3D Neuron Instance Segmentation in
 Any EM Volume with a Generalist Model" （[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/0518_paper.pdf))

 

# How does SegNeuron speed up neuron segmentation in EM volumes?
The general-purpose neuronal segmentation model, SegNeuron, achieves outstanding reconstruction performance on entirely unseen 3D EM datasets (x/y resolution: 5–10 nm). Human experts only need to perform connectivity corrections on the coarse segmentation results, which can then be directly used to fine-tune SegNeuron or to train new lightweight models. We are working to develop user-friendly tools based on Napari.
<p align="center">
  <img src="/Figures/pipeline.png"  alt="SegNeuron-based Pipeline" width="900"/>
</p>

## Environments
We have packaged all the dependencies into Connect.tar.gz, which can be directly downloaded for easy access [here](https://huggingface.co/yanchaoz/SegNeuron).
## Datasets and Models
The datasets required for model development and validation are available [here](https://huggingface.co/datasets/yanchaoz/EMNeuron). The trained models can be download [here](https://huggingface.co/yanchaoz/SegNeuron).
### Table: Details of EMNeuron

<div style="font-size: 0.6em;">
 
| Dataset              | Modality   | Res.($nm$) ($x,y,z$) | Total voxels (M) | Labeled voxels (M) | Dataset               | Modality   | Res.($nm$) ($x,y,z$) | Total voxels (M) | Labeled voxels (M) |
|----------------------|------------|----------------------|------------------|--------------------|-----------------------|------------|----------------------|------------------|--------------------|
| 1. ZFinch       | SBF-SEM    | 9, 9, 20             | 3635             | 131                | 9. HBrain         | FIB-SEM    | 8, 8, 8              | 3072             | 844                |
| 2. ZFish        | SBF-SEM    | 9, 9, 20             | 1674             | -                  | 10. FIB25        | FIB-SEM    | 8, 8, 8              | 312              | 312                |
| 3. _vEM1_            | ATUM-SEM   | 8, 8, 50             | 1205             | 157                | 11. Minnie       | ssTEM      | 8, 8, 40             | 2096             | -                  |
| 4. _vEM2_            | ATUM-SEM   | 8, 8, 30             | 1329             | 281                | 12. Pinky        | ssTEM      | 8, 8, 40             | 1165             | 117                |
| 5. _vEM3_            | ATUM-SEM   | 8, 8, 40             | 1301             | 253                | 13. FAFB         | ssTEM      | 8, 8, 40             | 2625             | 577                |
| 6. MitoEM        | ATUM-SEM   | 8, 8, 30             | 1048             | -                  | 14. Basil        | ssTEM      | 8, 8, 40             | 23               | 23                 |
| 7. H01           | ATUM-SEM   | 8, 8, 30             | 1166             | 118                | 15. Harris       | others     | 6, 6, 50             | 30               | 30                 |
| 8. Kasthuri      | ATUM-SEM   | 6, 6, 30             | 1526             | 478                | 16. _vEM4_            | others     | 8, 8, 20             | 45               | 45                 |


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

## Acknowledgement
This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf). Should you have any further questions, please let us know. Thanks again for your interest.
