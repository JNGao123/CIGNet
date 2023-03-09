# -*- coding: utf-8 -*-
#
#

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C


__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]

__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224
__C.CONST.IMG_H                             = 224
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 32
__C.CONST.N_VIEWS_RENDERING                 = 2                                                                                                         # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W                        = 128
__C.CONST.CROP_IMG_H                        = 128
__C.CONST.WEIGHTS                          = '/weight/CIGNet.pth'


__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'



__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True


__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = True
__C.TRAIN.NUM_WORKER                        = 4
__C.TRAIN.NUM_EPOCHES                       = 250
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER1_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER1_LEARNING_RATE             = 1e-3
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-3
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False



__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
