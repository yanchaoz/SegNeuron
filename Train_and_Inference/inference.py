import os
import yaml
import argparse
import imageio
import numpy as np
from attrdict import AttrDict
from collections import OrderedDict
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
from inference_provider import Provider_valid
from model.Mnet import MNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='SegNeuron', help='path to config file')
    args = parser.parse_args()
    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    pth = '/***/***.pth'

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU='sub').cuda()
    checkpoint = torch.load(pth)
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k.replace('module.', '') if 'module' in k else k
        new_state_dict[name] = v
    print('load mnet!')
    model.load_state_dict(new_state_dict)
    model = model.cuda()

    model.eval()
    valid_provider = Provider_valid(cfg, valid_data='***')
    criterion = nn.BCELoss()
    dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True)

    pbar = tqdm(total=len(valid_provider))
    losses_valid = []
    for k, batch in enumerate(dataloader, 0):
        inputs, target, _ = batch
        inputs = inputs.cuda()
        target = target.cuda()
        with torch.no_grad():
            pred, bound = model(inputs)
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        valid_provider.add_bound(np.squeeze(bound.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()

    out_affs = valid_provider.get_results()
    out_bounds = valid_provider.get_results_bound()

    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()
    valid_provider.reset_output()

    np.save('/***/***', out_affs)
    imageio.volwrite('/***/***.tif', out_bounds.squeeze())
