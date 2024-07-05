# SegNeuron
Official implementations, datasets and trained models for "SegNeuron: 3D Neuron Instance Segmentation in
 Any EM Volume with a Generalist Model"
## Dataset
The datasets required for model development and validation are as provided [here](https://huggingface.co/yanchaoz/EMNeuron).
## Model
The trained model can be download [here](https://huggingface.co/yanchaoz/SegNeuron).
https://huggingface.co/yanchaoz/SegNeuron
## Training
```python
cd /Pretrain
python pretrain.py
cd Train_and_Inference
python supervised_train.py
```
## Inference
```python
cd Train_and_Inference
python inference.py
cd Postprocess
python FRMC_post.py
```
## Acknowledgement
This code is based on SSNS-Net (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf]([http://example.com](https://github.com/constantinpape/elf). Should you have any further questions, please let us know. Thanks again for your interest.
