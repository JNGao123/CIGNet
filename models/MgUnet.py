from __future__ import absolute_import
import torch
import torch.nn as nn
from models.graph_conv import SemGraphConv
import numpy as np

class _GCN_conv3d_common(torch.nn.Module):
    def __init__(self, cfg,adj1,adj2,scale = 32, inputdim=1024,outputdim=1024,p_dropout=None,islastlayer = False):
        super(_GCN_conv3d_common, self).__init__()
        self.cfg = cfg
        self.dim = 2 * scale
        self.scale = scale
        self.batchsize =  cfg.CONST.BATCH_SIZE
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.view = cfg.CONST.N_VIEWS_RENDERING

        if(islastlayer):
            _gconv_input = [SemGraphConv(inputdim, outputdim, adj1)]

        else:
            _gconv_input = [ nn.Sequential(_GraphConv(adj1, self.inputdim , self.outputdim, p_dropout=p_dropout))]
         #   _gconv_input.append(_GraphNonLocal(1024))

        self.gconv_channel1 = nn.Sequential(*_gconv_input)

        if (islastlayer):
            _gconv_input1 = [SemGraphConv(inputdim, outputdim, adj2)]
        else:

            _gconv_input1 = [ nn.Sequential(_GraphConv(adj2, self.inputdim , self.outputdim, p_dropout=p_dropout))]
        ##    _gconv_input1.append(_GraphNonLocal(1024))

        self.gconv_channel2 = nn.Sequential(*_gconv_input1)

        if (islastlayer):
            _gconv_input2 = [SemGraphConv(inputdim, outputdim, adj2)]
        else:
            _gconv_input2 = [ nn.Sequential(_GraphConv(adj2, self.inputdim , self.outputdim, p_dropout=p_dropout))]
         ##   _gconv_input2.append(_GraphNonLocal(1024))

        self.gconv_channel3 = nn.Sequential(*_gconv_input2)


    def forward(self, coarse_volumes):
        coarse_volumes1 = coarse_volumes.permute(0, 1, 4, 2, 3).contiguous()
        coarse_volumes2 = coarse_volumes.permute(0, 1, 2, 3, 4).contiguous()
        coarse_volumes3 = coarse_volumes.permute(0, 1, 3, 2, 4).contiguous()


        views = self.view

        idx = np.random.permutation(self.view)
        while True:
            if(self.view==1):
                break
            for i in range(len(idx)):
                if(i==idx[i]):
                    idx = np.random.permutation(self.view)
                    break
                else:
                    continue;
            if(i==len(idx)-1):
                break
       # coarse_volumes1 = coarse_volumes1.permute(1, 0, 2, 3, 4).contiguous()
        coarse_volumes1_merge = list(torch.split(coarse_volumes1, 1,  dim=1))
        coarse_volumes1_merge_random = []
        for i in range(len(idx)):
            coarse_volumes1_merge_random.append(coarse_volumes1_merge[idx[i]])
        coarse_volumes1_merge_new = []
        for i in range(len(coarse_volumes1_merge)):
            coarse_volumes1_merge_new.append(torch.stack([coarse_volumes1_merge[i],coarse_volumes1_merge_random[i]],dim=1))
        merge1 = []
        for i in range(len(coarse_volumes1_merge)):
            coarse_volumes1temp = coarse_volumes1_merge_new[i].view((-1, self.dim, self.inputdim))
            coarse_volumes1temp = self.gconv_channel1(coarse_volumes1temp)
            coarse_volumes1temp = coarse_volumes1temp.view((-1,2, self.scale,  self.scale,  self.scale))
            coarse_volumes1temp = torch.mean(coarse_volumes1temp, dim=1)
            merge1.append(coarse_volumes1temp)
        merge1   =  torch.stack(merge1,dim=1)#.permute(1, 0, 2, 3, 4).contiguous()


        coarse_volumes2_merge = list(torch.split(coarse_volumes2, 1,  dim=1))
        coarse_volumes2_merge_random = []
        for i in range(len(idx)):
            coarse_volumes2_merge_random.append(coarse_volumes2_merge[idx[i]])
        coarse_volumes2_merge_new = []
        for i in range(len(coarse_volumes2_merge)):
            coarse_volumes2_merge_new.append(torch.stack([coarse_volumes2_merge[i],coarse_volumes2_merge_random[i]],dim=1))
        merge2 = []
        for i in range(len(coarse_volumes2_merge)):
            coarse_volumes2temp = coarse_volumes2_merge_new[i].view((-1, self.dim, self.inputdim))
            coarse_volumes2temp = self.gconv_channel2(coarse_volumes2temp)
            coarse_volumes2temp = coarse_volumes2temp.view((-1,2, self.scale,  self.scale,  self.scale))
            coarse_volumes2temp = torch.mean(coarse_volumes2temp, dim=1)
            merge2.append(coarse_volumes2temp)
        merge2  =  torch.stack(merge2,dim=1)#.permute(1, 0, 2, 3, 4).contiguous()

        coarse_volumes3_merge = list(torch.split(coarse_volumes3, 1,  dim=1))
        coarse_volumes3_merge_random = []
        for i in range(len(idx)):
            coarse_volumes3_merge_random.append(coarse_volumes3_merge[idx[i]])
        coarse_volumes3_merge_new = []
        for i in range(len(coarse_volumes3_merge)):
            coarse_volumes3_merge_new.append(torch.stack([coarse_volumes3_merge[i],coarse_volumes3_merge_random[i]],dim=1))
        merge3 = []
        for i in range(len(coarse_volumes3_merge)):
            coarse_volumes3temp = coarse_volumes3_merge_new[i].view((-1, self.dim, self.inputdim))
            coarse_volumes3temp = self.gconv_channel3(coarse_volumes3temp)
            coarse_volumes3temp = coarse_volumes3temp.view((-1,2, self.scale,  self.scale,  self.scale))
            coarse_volumes3temp = torch.mean(coarse_volumes3temp, dim=1)
            merge3.append(coarse_volumes3temp)
        merge3   =  torch.stack(merge3,dim=1)#.permute(1, 0, 2, 3, 4).contiguous()


        coarse_volumes1 = merge1.permute(0, 1, 3, 4, 2).contiguous()
        coarse_volumes2 = merge2.permute(0, 1, 2, 3, 4).contiguous()
        coarse_volumes3 = merge3.permute(0, 1, 3, 2, 4).contiguous()

        coarse_volumes_out =  coarse_volumes1*0.5+  coarse_volumes2*0.25+  coarse_volumes3*0.25


        return coarse_volumes_out


class _convDownSamping(torch.nn.Module):
    def __init__(self, cfg, channel,scale = 32):
        super(_convDownSamping, self).__init__()
        self.cfg = cfg
        self.scale = scale
        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, channel, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(channel),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(channel, 1,kernel_size=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()

        )

    def forward(self, coarse_volumes):
        coarse_volumes = coarse_volumes.view((-1, self.cfg.CONST.N_VIEWS_RENDERING,self.scale , self.scale , self.scale ))
        # image_features = torch.add(image_features, lanten)
        image_features = coarse_volumes.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)

        temp = []

        for features in image_features:
            volumes = features.view((-1, 1, self.scale , self.scale , self.scale ))
            # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
            volumes = self.layer1(volumes)
            volumes = self.layer2(volumes)

            temp.append(volumes)

        out = torch.cat(temp, dim=1)

        return out


class _convUpSamping(torch.nn.Module):
    def __init__(self, cfg, channel,scale=32):
        super(_convUpSamping, self).__init__()
        self.cfg = cfg
        self.scale = scale
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1, channel, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(channel),
            torch.nn.ReLU()
        )

        # Layer Definition
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(channel,1, kernel_size=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()

        )

    def forward(self, coarse_volumes):
        coarse_volumes = coarse_volumes.view((-1, self.cfg.CONST.N_VIEWS_RENDERING, self.scale , self.scale , self.scale ))
        # image_features = torch.add(image_features, lanten)
        image_features = coarse_volumes.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)

        temp = []

        for features in image_features:
            volumes = features.view((-1, 1, self.scale , self.scale , self.scale ))
            # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
            volumes = self.layer1(volumes)
            volumes = self.layer2(volumes)


            temp.append(volumes)

        out = torch.cat(temp, dim=1)

        return out


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class MgUnet(nn.Module):

    def __init__(self, cfg, adj1,adj2,adjsScale,hid_dim, coords_dim=(32 * 32, 32 * 32), num_layers=2, nodes_group=None,
                 p_dropout=None):
        super(MgUnet, self).__init__()

        self.cfg = cfg
        self.dim = cfg.CONST.N_VIEWS_RENDERING*32
        self.batchsize = cfg.CONST.BATCH_SIZE
        self.adj16=adjsScale[0]
        self.adjnosys16 =adjsScale[1]
        self.adj8 =adjsScale[2]
        self.adjnosys8 =adjsScale[3]
        self.adj4 =adjsScale[4]
        self.adjnosys4 =adjsScale[5]
        self.cenDim = 64


        self.gconv_layers1 = nn.Sequential(_GCN_conv3d_common(cfg, adj1, adj2))
        self.gconv_layersC3D_16 = nn.Sequential(_convDownSamping(cfg,  self.cenDim ))


        self.gconv_layers2 = nn.Sequential(_GCN_conv3d_common(cfg, self.adj16, self.adjnosys16,scale = 16,inputdim=16*16,outputdim=16*16))
        self.gconv_layersC3D_8 = nn.Sequential(_convDownSamping(cfg,  self.cenDim ,scale=16))

        self.gconv_layers3 = nn.Sequential(_GCN_conv3d_common(cfg,self.adj8, self.adjnosys8,scale =8,inputdim=8*8,outputdim=8*8))
        self.gconv_layersC3D_4 = nn.Sequential(_convDownSamping(cfg,  self.cenDim ,scale=8))

        self.gconv_layers4 = nn.Sequential(_GCN_conv3d_common(cfg, self.adj4, self.adjnosys4,scale =4,inputdim=4*4,outputdim=4*4))
        self.gconv_layersC3U_8 = nn.Sequential(_convUpSamping(cfg,  self.cenDim ,scale=4))

        self.gconv_layers5 = nn.Sequential(_GCN_conv3d_common(cfg, self.adj8, self.adjnosys8,scale =8,inputdim=8*8,outputdim=8*8))

        self.gconv_layersC3U_16 = nn.Sequential(_convUpSamping(cfg,  self.cenDim ,scale=8))
        self.gconv_layers6 = nn.Sequential(_GCN_conv3d_common(cfg, self.adj16, self.adjnosys16,scale =16,inputdim=16*16,outputdim=16*16))

        self.gconv_layersC3U_32 = nn.Sequential(_convUpSamping(cfg,  self.cenDim ,scale=16))
        self.gconv_layers7 = nn.Sequential(_GCN_conv3d_common(cfg, adj1, adj2,islastlayer=True))


        self.sigmodlayer = nn.Sequential(torch.nn.Sigmoid())

    def forward(self, x):

        out1 =  self.gconv_layers1(x)

        out_d16 =self.gconv_layersC3D_16(out1)
        out2 = self.gconv_layers2(out_d16)
        out_d8 = self.gconv_layersC3D_8(out2)
        out3 = self.gconv_layers3(out_d8)
        out_d4 = self.gconv_layersC3D_4(out3)

        out4 = self.gconv_layers4(out_d4)
        out_u8 = self.gconv_layersC3U_8(out4)

        out5 = self.gconv_layers5(out_u8+out3)

        out_u16 = self.gconv_layersC3U_16(out5)

        out6 = self.gconv_layers6(out_u16+out2)
        out_u32 = self.gconv_layersC3U_32(out6)


        out7 = self.gconv_layers7(out_u32+out1)
        out = self.sigmodlayer(out7)
        out = out.view((-1, self.cfg.CONST.N_VIEWS_RENDERING, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
        out = torch.mean(out, dim=1)

        out = out.permute(0, 3, 1, 2).contiguous()


        im_features = torch.split(out, 1, dim=1)

        im_symmetry1 = []
        im_symmetry2 = []
        for i in range(16):
            f1 = im_features[15 - i]
            f2 = im_features[16 + i]
            im_symmetry1.append(f1.squeeze(dim=1))
            im_symmetry2.append(f2.squeeze(dim=1))
        im_symmetry1 = torch.stack(im_symmetry1).permute(1, 0, 2,3).contiguous()
        im_symmetry2 = torch.stack(im_symmetry2).permute(1, 0, 2,3).contiguous()

        output = out.permute(0, 2, 3,1).contiguous()


        return output, im_symmetry1,im_symmetry2



