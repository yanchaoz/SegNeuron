from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch
import torch.nn as nn
from pretrain_provider import Provider
from utils.show import show_bound
from model.Mnet_pretrain import MNet
from utils.utils import setup_seed
import torch.nn.functional as F


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    print('load mnet!')

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU='sub').cuda()

    if cfg.MODEL.pre_train:
        ckpt_path = cfg.MODEL.pretrain_path
        print('Load pre-trained model from' + ckpt_path)
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module' in k else k
            pretrained_dict[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict}  # 1. filter out unnecessary keys
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)  # 2. overwrite entries in the existing state dict
        model.load_state_dict(model_dict)

    if cfg.MODEL.continue_train:
        ckpt_path = cfg.MODEL.continue_path
        print('Load pre-trained model from' + ckpt_path)
        checkpoint = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module' in k else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = model.to(device)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, model, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_labeled_loss = 0
    sum_unlabel_loss = 0

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        inputs, gt, hog = train_provider.next()

        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()

        pred, pred_hog = model(inputs)
        # LOSS
        ##############################
        loss1 = F.mse_loss(pred, gt)
        loss2 = F.mse_loss(pred_hog, hog)
        loss = 0.2 * loss1 + loss2
        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info(
                    'step %d, loss = %.6f, labeled_loss=%.6f, unlabel_loss=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                    % (iters, sum_loss, sum_labeled_loss, sum_unlabel_loss, current_lr, sum_time,
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info(
                    'step %d, loss = %.6f, labeled_loss=%.6f, unlabel_loss=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' \
                    % (iters, sum_loss / cfg.TRAIN.display_freq * 1, \
                       sum_labeled_loss / cfg.TRAIN.display_freq * 1, \
                       sum_unlabel_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time, \
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = %d, loss = %.6f, labeled_loss=%.6f, unlabel_loss=%.6f' \
                             % (iters, sum_loss / cfg.TRAIN.display_freq * 1, \
                                sum_labeled_loss / cfg.TRAIN.display_freq * 1, \
                                sum_unlabel_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_bound(iters, inputs, pred, gt, cfg.cache_path, model_type=cfg.MODEL.model_type)

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='SegNeuron', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                     eps=0.01, weight_decay=1e-6, amsgrad=True)
        init_iters = 0
        loop(cfg, train_provider, model, optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')