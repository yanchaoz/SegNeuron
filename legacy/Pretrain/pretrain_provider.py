from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.feature import hog
from utils.augmentation import SimpleAugment as Filp
import imageio
from einops.layers.torch import Rearrange
from einops import repeat, rearrange


class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        self.cfg = cfg
        self.simple_aug = Filp()
        self.crop_from_origin = [20, 128, 128]
        self.dataset = []
        self.labels = []
        dataset_list = ['J0126-sbem', 'Kasthuri-atum', 'Hemi-brain-fib', 'CREMI-sstem', 'AxonEM[M]-sstem',
                        'AxonEM[H]-atum', 'Mira-adwt', 'Fib-25-fib', 'Mira-scn', 'Mira-fish', 'MitoEM-atum',
                        'Minnie-sstem', 'Zfish-sbfsem']

        for sub_path in dataset_list:
            self.folder_name = os.path.join(cfg.DATA.data_folder, sub_path)
            file_num = len(os.listdir(self.folder_name))
            train_datasets = ['%d.tif' % i for i in range(file_num)]
            for k in range(len(train_datasets)):
                print('load ' + self.folder_name + train_datasets[k] + ' ...')
                data = imageio.volread(os.path.join(self.folder_name, train_datasets[k]))
                self.dataset.append(data[:])

    def __getitem__(self, index):

        k = random.randint(0, len(self.dataset))
        used_data = self.dataset[k]
        raw_data_shape = used_data.shape

        random_z = random.randint(0, raw_data_shape[0] - self.crop_from_origin[0])
        random_y = random.randint(0, raw_data_shape[1] - self.crop_from_origin[1])
        random_x = random.randint(0, raw_data_shape[2] - self.crop_from_origin[2])

        imgs1 = used_data[random_z:random_z + self.crop_from_origin[0], \
                random_y:random_y + self.crop_from_origin[1], \
                random_x:random_x + self.crop_from_origin[2]].copy()

        [imgs1] = self.simple_aug([imgs1])
        noise = np.random.normal(loc=np.mean(imgs1), scale=np.std(imgs1),
                                 size=(self.crop_from_origin[0], self.crop_from_origin[1], self.crop_from_origin[2]))

        imgs1 = imgs1.astype(np.float32) / 255.0
        noise = noise.astype(np.float32) / 255.0

        # HOG feature
        pixels_per_cell = (4, 4)
        cells_per_block = (1, 1)

        hog_stack = []
        for slice in range(self.crop_from_origin[0]):
            _, hog_image = hog(imgs1[slice], pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                               visualize=True)
            hog_stack.append(hog_image)
        hog_stack = np.array(hog_stack)

        # Mask
        mask = np.ones_like(imgs1)
        z_bx = random.choice([2, 4, 5])
        x_y_bx = random.choice([4, 8, 16])
        mask_ratio = 0.5 + 0.2 * random.random()

        to_patch = Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=x_y_bx, p2=x_y_bx, pf=z_bx)
        to_img = Rearrange('b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=int(20 / z_bx), h=int(128 / x_y_bx),
                           w=int(128 / x_y_bx), p1=x_y_bx, p2=x_y_bx, pf=z_bx, c=1)

        patches = to_patch(torch.tensor(mask.reshape(1, 1, 20, 128, 128)))
        patches = rearrange(patches, 'b p (h w d) -> b p h w d', h=x_y_bx, w=x_y_bx, d=z_bx)
        random_dimensions = np.random.choice(int(20 / z_bx) * int(128 / x_y_bx) * int(128 / x_y_bx), size=int(
            int(20 / z_bx) * int(128 / x_y_bx) * int(128 / x_y_bx) * mask_ratio), replace=False)
        patches[:, random_dimensions, :, :, :] = 0

        patches = rearrange(patches, 'b p h w d -> b p (h w d)', h=x_y_bx, w=x_y_bx, d=z_bx)
        patches = to_img(patches)
        mask = patches.numpy().squeeze()

        imgs_mask = mask * imgs1 + (1 - mask) * noise

        imgs1 = imgs1[np.newaxis, ...]
        imgs1 = np.ascontiguousarray(imgs1, dtype=np.float32)

        imgs_mask = imgs_mask[np.newaxis, ...]
        imgs_mask = np.ascontiguousarray(imgs_mask, dtype=np.float32)

        hog_stack = hog_stack[np.newaxis, ...]
        hog_stack = np.ascontiguousarray(hog_stack, dtype=np.float32)

        return imgs_mask, imgs1, hog_stack

    def __len__(self):
        return int(sys.maxsize)


class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(
                DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
            return batch
