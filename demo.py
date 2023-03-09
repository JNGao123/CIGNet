# -*- coding: utf-8 -*-
#

import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from models.encoder import Encoder
from models.decoder import Decoder
from models.CifFusion import CifFusion
from models.decoder_ced import DecoderCED
from models.MgUnet import MgUnet
from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint
from config import cfg
import matplotlib
import numpy as np
import os
from models.graph_utils import adj_mx_from_skeleton1,adj_mx_from_skeleton2,gatScaleAdj

matplotlib.use('Agg')


def load_model():

    encoder_ied = Encoder(cfg)
    decoder_ied = Decoder(cfg)
    encoder_ced = Encoder(cfg)
    decoder_ced = DecoderCED(cfg)
    mg_unet = MgUnet(cfg, adj_mx_from_skeleton1(cfg), adj_mx_from_skeleton2(cfg), gatScaleAdj(cfg), 32 * 32)
    cif_fusion = CifFusion(cfg)

    if torch.cuda.is_available():
        encoder_ied = torch.nn.DataParallel(encoder_ied).cuda()
        decoder_ied = torch.nn.DataParallel(decoder_ied).cuda()
        encoder_ced = torch.nn.DataParallel(encoder_ced).cuda()
        decoder_ced = torch.nn.DataParallel(decoder_ced).cuda()
        mg_unet = torch.nn.DataParallel(mg_unet).cuda()
        cif_fusion = torch.nn.DataParallel(cif_fusion).cuda()

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    encoder_ied.load_state_dict(checkpoint['encoder_state_dict'])
    decoder_ied.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_ced.load_state_dict(checkpoint['encoder1_state_dict'])
    decoder_ced.load_state_dict(checkpoint['decoder1_state_dict'])
    if cfg.NETWORK.USE_REFINER:
        mg_unet.load_state_dict(checkpoint['refiner_state_dict'])
    if cfg.NETWORK.USE_MERGER:
        cif_fusion.load_state_dict(checkpoint['merger_state_dict'])

    return encoder_ied,decoder_ied,encoder_ced,decoder_ced,cif_fusion,mg_unet

def test_net(rendering_images,
             ground_truth_volume,
             encoder_ied=None,
             decoder_ied=None,
             encoder_ced=None,
             decoder_ced=None,
             mg_unet=None,
             cif_fusion=None):



    # Switch models to evaluation mode
    encoder_ied.eval()
    decoder_ied.eval()
    encoder_ced.eval()
    decoder_ced.eval()
    mg_unet.eval()
    cif_fusion.eval()



    with torch.no_grad():
        # Get data from data loader
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)
        ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

        #get local and global feature
        image_features = encoder_ied(rendering_images)
        raw_features, generated_volume = decoder_ied(image_features)

        image_features1 = encoder_ced(rendering_images)
        raw_features1, generated_volume1,classfea= decoder_ced(image_features1)


        if cfg.NETWORK.USE_MERGER:
            generated_volume = cif_fusion(raw_features, generated_volume, raw_features1, generated_volume1)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)

        if cfg.NETWORK.USE_REFINER:
            generated_volume, im_symmetry1, im_symmetry2 = mg_unet(generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)


        # IoU per sample
        sample_iou = []
        for th in cfg.TEST.VOXEL_THRESH:
            _volume = torch.ge(generated_volume, th).float()
            intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
            union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
            sample_iou.append((intersection / union).item())



    return sample_iou





def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test-1view-gcn.txt', dest='test-1view-gcn.txt', help='Test neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def main():

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    test_net(cfg)


if __name__ == '__main__':
    import cv2
    img_path = "/home/gaojunna/paper2020GJN-TheSecondIdear/paper_1_git/CIGNet/testdata/fff199c067a6e0f019fb4103277a6b93/rendering/"
    img_names = ["01.png","02.png"]
    vox_path = "/home/gaojunna/paper2020GJN-TheSecondIdear/paper_1_git/CIGNet/testdata/fff199c067a6e0f019fb4103277a6b93/"
    vox_name = "model.binvox"
    rendering_images = [cv2.imread(img_path+im, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255. for im in img_names]

    with open(vox_path+vox_name, 'rb') as f:
        volume = utils.binvox_rw.read_as_3d_array(f)
        volume = volume.data.astype(np.float32)


    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    rendering_images = test_transforms(rendering_images)
    rendering_images = torch.from_numpy(np.array([np.asarray(rendering_images)]))
    volume = torch.from_numpy(np.array([volume]))

    encoder_ied, decoder_ied, encoder_ced, decoder_ced, cif_fusion, mg_unet = load_model()

    ious = test_net(rendering_images=rendering_images,
             ground_truth_volume = volume,
             encoder_ied = encoder_ied,
             decoder_ied = decoder_ied,
             encoder_ced = encoder_ced,
             decoder_ced = decoder_ced,
             cif_fusion = cif_fusion,
             mg_unet = mg_unet)


    print(ious)
