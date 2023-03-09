# -*- coding: utf-8 -*-
#
#

import torch
import numpy as np
from datetime import datetime as dt

def label1(label):
    size = len(label)
    labelList = []

    for i in range(size):
        idx = label[i]
        if idx == "02691156":
            labelList.append(0)
        elif idx == "02828884":
            labelList.append(1)
        elif idx == "02933112":
            labelList.append(2)
        elif idx == "02958343":
            labelList.append(3)
        elif idx == "03001627":
            labelList.append(4)
        elif idx == "03211117":
            labelList.append(5)
        elif idx == "03636649":
            labelList.append(6)
        elif idx == "03691459":
            labelList.append(7)
        elif idx == "04090263":
            labelList.append(8)
        elif idx == "04256520":
            labelList.append(9)
        elif idx == "04379243":
            labelList.append(10)
        elif idx == "04401088":
            labelList.append(11)
        elif idx == "04530566":
            labelList.append(12)
    labels = np.array(labelList, dtype='int')
    return torch.from_numpy(labels)

def label(label):
    size = len(label)
    labels = np.zeros((size,13),dtype = 'float32')

    for i in range(size):
        idx = label[i]
        if idx == "02691156":
            labels[i, 0] = 1
        elif idx == "02828884":
            labels[i, 1] = 1
        elif idx == "02933112":
            labels[i, 2] = 1
        elif idx == "02958343":
            labels[i, 3] = 1
        elif idx == "03001627":
            labels[i, 4] = 1
        elif idx == "03211117":
            labels[i, 5] = 1
        elif idx == "03636649":
            labels[i, 6] = 1
        elif idx == "03691459":
            labels[i, 7] = 1
        elif idx == "04090263":
            labels[i, 8] = 1
        elif idx == "04256520":
            labels[i, 9] = 1
        elif idx == "04379243":
            labels[i, 10] = 1
        elif idx == "04401088":
            labels[i, 11] = 1
        elif idx == "04530566":
            labels[i, 12] = 1

    return torch.from_numpy(labels)


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, encoder, encoder_solver, decoder, decoder_solver,  encoder1, encoder_solver1, decoder1, decoder_solver1,refiner,
                     refiner_solver, merger, merger_solver, best_iou, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'decoder_solver_state_dict': decoder_solver.state_dict(),
        'encoder1_state_dict': encoder1.state_dict(),
        'encoder1_solver_state_dict': encoder_solver1.state_dict(),
        'decoder1_state_dict': decoder1.state_dict(),
        'decoder1_solver_state_dict': decoder_solver1.state_dict()
    }

    if cfg.NETWORK.USE_REFINER:
        checkpoint['refiner_state_dict'] = refiner.state_dict()
        checkpoint['refiner_solver_state_dict'] = refiner_solver.state_dict()
    if cfg.NETWORK.USE_MERGER:
        checkpoint['merger_state_dict'] = merger.state_dict()
        checkpoint['merger_solver_state_dict'] = merger_solver.state_dict()

    torch.save(checkpoint, file_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
