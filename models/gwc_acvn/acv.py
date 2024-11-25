from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.gwc_acvn.submodule import *
import math
import gc
import time


class feature_extraction(nn.Module):
    def __init__(self, concat_feature_channel=32):
        super(feature_extraction, self).__init__()
        
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.firstconv(x)
        x2 = self.layer1(x1)
        l2 = self.layer2(x2)
        l3 = self.layer3(l2)

        l4 = self.layer4(l3)
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        concat_feature = self.lastconv(gwc_feature)
        return {"gwc_feature": gwc_feature, "concat_feature": concat_feature, "early_feats1": x1, "early_feats2": x2}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)


        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class ACVNet(nn.Module):
    def __init__(self, maxdisp, loss_config):
        super(ACVNet, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 40
        self.concat_channels = 32
        self.transform_feats = False

        self.loss_config = loss_config

        self.feature_extraction = feature_extraction(concat_feature_channel=self.concat_channels)

        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                  bias=False))
        self.patch = nn.Conv3d(40, 40, kernel_size=(1,3,3), stride=1, dilation=1, groups=40, padding=(0,1,1), bias=False)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, dilation=1, groups=8, padding=(0,1,1), bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=2, groups=16, padding=(0,2,2), bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=3, groups=16, padding=(0,3,3), bias=False)

        self.dres1_att = nn.Sequential(convbn_3d(40, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1)) 
        self.dres2_att = hourglass(16)
        self.classif_att = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))


        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))        

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right, teaching=False, att_weights=False):
        self.return_att_weights = att_weights
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        gwc_patch_volume= self.patch(gwc_volume) # >>>>>>> Gwc-p <<<<<<<<<<<<
        patch_l1 = self.patch_l1(gwc_patch_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_patch_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_patch_volume[:, 24:40])
        patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1) # >>>>>>> Gwc-mp <<<<<<<<<<<<
        #40 48 64 128
        cost_attention = self.dres1_att(patch_volume)
        cost_attention = self.dres2_att(cost_attention) # >>>>>>> Gwc-mp-hg <<<<<<<<<<<<
        att_weights = self.classif_att(cost_attention)
        # ac_volume = att_weights * concat_volume
        # print(ac_volume.shape)
        # print(gwc_volume.shape)

        concat_feature_left = self.concatconv(features_left["gwc_feature"])
        concat_feature_right = self.concatconv(features_right["gwc_feature"])
        concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.maxdisp // 4)
        att_weights_soft = F.softmax(att_weights, dim=2)
        ac_volume = att_weights_soft * concat_volume  ### ac_volume = att_weights * concat_volume  # >>>>>>> Gwc-mp-att <<<<<<<<<<<<
        #64 48 64 128
        cost01 = self.dres0(ac_volume)
        out0 = self.dres1(cost01) + cost01
        out1 = self.dres2(out0)
        out2 = self.dres3(out1)

        if self.training:

            cost0x = self.classif0(cost01)
            cost1x = self.classif1(out1)
            cost2x = self.classif2(out2)

            cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost_attention = torch.squeeze(cost_attention, 1)
            pred_attention = F.softmax(cost_attention, dim=1)
            pred_attention = disparity_regression(pred_attention, self.maxdisp) # >>>>>>> Gwc-mp-att-hg-s <<<<<<<<<<<<

            cost0 = F.upsample(cost0x, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1x, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2x, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            return [pred_attention, pred0, pred1, pred2]

        else:

            # cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            # cost_attention = torch.squeeze(cost_attention, 1)
            # pred_attention = F.softmax(cost_attention, dim=1)
            # pred_attention = disparity_regression(pred_attention, self.maxdisp)

            cost2 = self.classif2(out2)
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            if teaching:

                feat_outputs = []
                soft_outputs = []

                if "early_feats" in self.loss_config['feat_config']:
                    feat_outputs.extend([features_left["early_feats1"], features_left["early_feats2"], \
                                         features_right["early_feats1"], features_right["early_feats2"]])

                if "2feats" in self.loss_config['feat_config']:
                    feat_outputs.extend([features_left["gwc_feature"], features_right["gwc_feature"]])

                if "gwc_vol" in self.loss_config['feat_config'] or "att_vol" in self.loss_config['feat_config']:
                    feat_outputs.extend([gwc_volume])

                if "gwc_patch_vol" in self.loss_config['feat_config']:
                    feat_outputs.extend([gwc_patch_volume])

                if 'hg_vols' in self.loss_config['feat_config']:
                    feat_outputs.extend([out0, out1, out2])

                if 'att_weights' in self.loss_config['feat_config']:
                    feat_outputs.extend([att_weights_soft])

                if 'costx_vols' in self.loss_config['feat_config']:
                    cost0x = self.classif0(cost01)
                    cost1x = self.classif1(out1)
                    cost2x = self.classif2(out2)
                    feat_outputs.extend([cost0x, cost1x, cost2x])

                if 'all_tea_costs' in self.loss_config['soft_config']:

                    cost1x = self.classif1(out1)
                    cost1 = F.upsample(cost1x, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                    cost1 = torch.squeeze(cost1, 1)

                    cost0x = self.classif0(cost01)
                    cost0 = F.upsample(cost0x, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                    cost0 = torch.squeeze(cost0, 1)

                    soft_outputs.extend([cost0, cost1])

                if 'cost1' in self.loss_config['soft_config']:
                    cost1x = self.classif1(out1)
                    cost1 = F.upsample(cost1x, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                    cost1 = torch.squeeze(cost1, 1)
                    soft_outputs.extend([cost1])

                if 'cost2' in self.loss_config['soft_config']:
                    soft_outputs.extend([cost2])

                return [pred2], feat_outputs, soft_outputs

                # TODO att
                # elif self.loss_type == "gwc_vol" or "cost_plus_att":
                #     if self.return_att_weights:
                #         return [pred2], gwc_volume, att_weights
                #     return [pred2], gwc_volume1

            # return [pred_attention, pred2]
            return [pred2]




def ACV(d, loss_config):
    return ACVNet(d, loss_config)
