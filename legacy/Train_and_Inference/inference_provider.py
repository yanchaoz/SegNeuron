import os
import cv2
import h5py
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph
import imageio


class Provider_valid(Dataset):
    def __init__(self, cfg, valid_data=None):
        # basic settings
        self.cfg = cfg
        if valid_data is not None:
            valid_dataset_name = valid_data
        else:
            valid_dataset_name = cfg.DATA.valid_dataset
        print('valid dataset:', valid_dataset_name)

        self.crop_size = [20, 128, 128]
        self.stride = [10, 64, 64]
        self.out_size = [20, 128, 128]

        if valid_dataset_name == 'Harris':
            self.sub_path = 'OutofDistribution/Harris'
            self.train_datasets = ['raw.tif']
            self.train_labels = ['label.tif']
        elif valid_dataset_name == 'Microns[B]':
            self.sub_path = 'OutofDistribution/Microns[B]'
            self.train_datasets = ['raw.tif']
            self.train_labels = ['label.tif']
        elif valid_dataset_name == 'IONSEM':
            self.sub_path = 'OutofDistribution/Mira-ionsem'
            self.train_datasets = ['raw.tif']
            self.train_labels = ['label.tif']
        else:
            raise AttributeError('No this dataset type!')

        self.folder_name = os.path.join(cfg.DATA.data_folder_val, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        # load dataset
        self.dataset = []
        self.labels = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.folder_name + self.train_datasets[k] + ' ...')
            data = imageio.volread(os.path.join(self.folder_name, self.train_datasets[k]))
            self.dataset.append(data[:])
            label = imageio.volread(os.path.join(self.folder_name, self.train_labels[k]))
            self.labels.append(label[:])
        self.origin_data_shape = list(self.dataset[0].shape)

        self.gt_affs = []
        for k in range(len(self.labels)):
            temp = self.labels[k].copy()
            self.gt_affs.append(seg_to_affgraph(temp, mknhood3d(1), pad='replicate').astype(np.float32))

        self.num_zyx = [(self.origin_data_shape[0] - self.out_size[0]) // self.stride[0] + 2,
                        (self.origin_data_shape[1] - self.out_size[1]) // self.stride[1] + 2,
                        (self.origin_data_shape[2] - self.out_size[2]) // self.stride[2] + 2]
        self.valid_padding = [
            ((self.num_zyx[0] - 1) * self.stride[0] + self.out_size[0] - self.origin_data_shape[0]) // 2,
            ((self.num_zyx[1] - 1) * self.stride[1] + self.out_size[1] - self.origin_data_shape[1]) // 2,
            ((self.num_zyx[2] - 1) * self.stride[2] + self.out_size[2] - self.origin_data_shape[2]) // 2]

        for k in range(len(self.dataset)):
            self.dataset[k] = np.pad(self.dataset[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                       (self.valid_padding[1], self.valid_padding[1]), \
                                                       (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
            self.labels[k] = np.pad(self.labels[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                     (self.valid_padding[1], self.valid_padding[1]), \
                                                     (self.valid_padding[2], self.valid_padding[2])), mode='reflect')

        # the training dataset size
        self.raw_data_shape = list(self.dataset[0].shape)
        print('valid_size:', self.raw_data_shape)

        self.reset_output()
        self.weight_vol = self.get_weight()

        # the number of inference times
        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset * len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        pos_data = index // self.num_per_dataset
        pre_data = index % self.num_per_dataset
        pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
        pos_x = pos_xy // self.num_zyx[2]
        pos_y = pos_xy % self.num_zyx[2]

        # find position
        fromz = pos_z * self.stride[0]
        endz = fromz + self.crop_size[0]
        if endz > self.raw_data_shape[0]:
            endz = self.raw_data_shape[0]
            fromz = endz - self.crop_size[0]
        fromy = pos_y * self.stride[1]
        endy = fromy + self.crop_size[1]
        if endy > self.raw_data_shape[1]:
            endy = self.raw_data_shape[1]
            fromy = endy - self.crop_size[1]
        fromx = pos_x * self.stride[2]
        endx = fromx + self.crop_size[2]
        if endx > self.raw_data_shape[2]:
            endx = self.raw_data_shape[2]
            fromx = endx - self.crop_size[2]

        self.pos = [fromz, fromy, fromx]

        imgs = self.dataset[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()
        lb = self.labels[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()
        lb_affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)

        weight_factor = np.sum(lb_affs) / np.size(lb_affs)
        weight_factor = np.clip(weight_factor, 1e-3, 1)
        weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)

        imgs = imgs.astype(np.float32) / 255.0
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        return imgs, lb_affs, weightmap

    def __len__(self):
        return self.iters_num

    def reset_output(self):
        self.out_affs = np.zeros(tuple([3] + self.raw_data_shape), dtype=np.float32)
        self.weight_map = np.zeros(tuple([1] + self.raw_data_shape), dtype=np.float32)

        self.out_affs2 = np.zeros(tuple([1] + self.raw_data_shape), dtype=np.float32)
        self.weight_map2 = np.zeros(tuple([1] + self.raw_data_shape), dtype=np.float32)

        self.out_affs3 = np.zeros(tuple([1] + self.raw_data_shape), dtype=np.float32)
        self.weight_map3 = np.zeros(tuple([1] + self.raw_data_shape), dtype=np.float32)

    def get_weight(self, sigma=0.2, mu=0.0):
        zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                 np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                 np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def add_vol(self, affs_vol):
        fromz, fromy, fromx = self.pos
        self.out_affs[:, fromz:fromz + self.out_size[0], \
        fromx:fromx + self.out_size[1], \
        fromy:fromy + self.out_size[2]] += affs_vol * self.weight_vol
        self.weight_map[:, fromz:fromz + self.out_size[0], \
        fromx:fromx + self.out_size[1], \
        fromy:fromy + self.out_size[2]] += self.weight_vol

    def get_results(self):
        self.out_affs = self.out_affs / self.weight_map
        self.out_affs = self.out_affs[:, self.valid_padding[0]:-self.valid_padding[0], \
                        self.valid_padding[1]:-self.valid_padding[1], \
                        self.valid_padding[2]:-self.valid_padding[2]]

        return self.out_affs

    def add_bound(self, affs_vol):
        fromz, fromy, fromx = self.pos
        self.out_affs2[:, fromz:fromz + self.out_size[0], \
        fromx:fromx + self.out_size[1], \
        fromy:fromy + self.out_size[2]] += affs_vol * self.weight_vol
        self.weight_map2[:, fromz:fromz + self.out_size[0], \
        fromx:fromx + self.out_size[1], \
        fromy:fromy + self.out_size[2]] += self.weight_vol

    def get_results_bound(self):
        self.out_affs2 = self.out_affs2 / self.weight_map2
        self.out_affs2 = self.out_affs2[:, self.valid_padding[0]:-self.valid_padding[0], \
                         self.valid_padding[1]:-self.valid_padding[1], \
                         self.valid_padding[2]:-self.valid_padding[2]]

        return self.out_affs2

    def get_gt_affs(self, num_data=0):
        return self.gt_affs[num_data].copy()

    def get_gt_lb(self, num_data=0):
        lbs = self.labels[num_data].copy()
        return lbs[self.valid_padding[0]:-self.valid_padding[0], \
               self.valid_padding[1]:-self.valid_padding[1], \
               self.valid_padding[2]:-self.valid_padding[2]]

    def get_raw_data(self, num_data=0):
        out = self.dataset[num_data].copy()
        return out[self.valid_padding[0]:-self.valid_padding[0], \
               self.valid_padding[1]:-self.valid_padding[1], \
               self.valid_padding[2]:-self.valid_padding[2]]
