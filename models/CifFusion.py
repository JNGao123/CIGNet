# -*- coding: utf-8 -*-
#
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CifFusion(torch.nn.Module):
    def __init__(self, cfg):
        super(CifFusion, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(4),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

        self.layer11 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer22 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer33 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(4),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer44 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer55 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

    def forward(self, raw_features, coarse_volumes, raw_features1, coarse_volumes1):
        n_views_rendering = coarse_volumes.size(1)
        raw_features = torch.split(raw_features, 1, dim=1)
        raw_features1 = torch.split(raw_features1, 1, dim=1)

        coarse_volumes = torch.split(coarse_volumes, 1, dim=1)
        coarse_volumes1 = torch.split(coarse_volumes1, 1, dim=1)

        volume_fin = []
        out = []
        for i in range(n_views_rendering):
            volume = []
            volume_weights = []
            raw_feature = torch.squeeze(raw_features[i], dim=1)

            volume_weight = self.layer1(raw_feature)

            volume_weight = self.layer2(volume_weight)

            volume_weight = self.layer3(volume_weight)

            volume_weight = self.layer4(volume_weight)

            volume_weight = self.layer5(volume_weight)

            volume_weight = torch.squeeze(volume_weight, dim=1)

            raw_feature1 = torch.squeeze(raw_features1[i], dim=1)

            volume_weight1 = self.layer11(raw_feature1)

            volume_weight1 = self.layer22(volume_weight1)

            volume_weight1 = self.layer33(volume_weight1)

            volume_weight1 = self.layer44(volume_weight1)

            volume_weight1 = self.layer55(volume_weight1)

            volume_weight1 = torch.squeeze(volume_weight1, dim=1)

            volume_weights.append(volume_weight)
            volume_weights.append(volume_weight1)

            volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
            volume_weights = torch.softmax(volume_weights, dim=1)
            out.append(volume_weights.squeeze())

            volume.append(torch.squeeze(coarse_volumes[i], dim=1) )
            volume.append(torch.squeeze(coarse_volumes1[i], dim=1))

            volume = torch.stack(volume).permute(1, 0, 2, 3, 4).contiguous()

            out.append(volume.squeeze())

            volume = volume * volume_weights
            volume = torch.sum(volume, dim=1)

            volume_fin.append(volume)

        volume_fin = torch.stack(volume_fin).permute(1, 0, 2, 3, 4).contiguous()

        return torch.clamp(volume_fin, min=0, max=1)
