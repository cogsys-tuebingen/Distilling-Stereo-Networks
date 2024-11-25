from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.gwc_submodule import *
import math


class feature_extraction(nn.Module):
    def __init__(self, trans_feats=False, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature
        self.trans_feats = trans_feats

        self.inplanes = 32
        self.firstconv = nn.Sequential(ConvBN(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       ConvBN(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       ConvBN(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 2, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 4, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 1, 1, 1, 2)


        if self.trans_feats:
            self.feat_trans = nn.Sequential(ConvBN(320, 120, 1, 1, 0, 1),
                                            nn.LeakyReLU(inplace=True),
                                            ConvBN(120, 32, 1, 1, 0, 1))

        if self.concat_feature:
            self.lastconv = nn.Sequential(ConvBN(320, 128, 3, 1, 1, 1),
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

        # if self.trans_feats:
        #     print("ASDASDASDASD")
        #     trans_feature = self.feat_trans(gwc_feature)
        #     return {"gwc_feature": gwc_feature, "trans_feature": trans_feature}

        return {"gwc_feature": gwc_feature, "early_feats1": x1, "early_feats2": x2}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(ConvBN_3D(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(ConvBN_3D(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(ConvBN_3D(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(ConvBN_3D(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = ConvBN_3D(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBN_3D(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class GwcNet(nn.Module):
    def __init__(self, maxdisp, loss_config, transform_feats=False, use_concat_volume=False):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.loss_config = loss_config
        self.transform_feats = True

        self.num_groups = 40
        self.hg_size = 16

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(self.transform_feats, concat_feature=False)


        self.attention_weights = nn.Parameter(torch.zeros((1, 48, 64, 128), dtype=torch.float32, requires_grad=True))

        if 'hg_vols' in self.loss_config['feat_config']:
            self.cost_trans32a =  nn.Sequential(ConvBN_3D(self.hg_size, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      ConvBN_3D(32, 32, 3, 1, 1))
            self.cost_trans32b =  nn.Sequential(ConvBN_3D(self.hg_size, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      ConvBN_3D(32, 32, 3, 1, 1))
            self.cost_trans32c =  nn.Sequential(ConvBN_3D(self.hg_size, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      ConvBN_3D(32, 32, 3, 1, 1))

            # nn.init.kaiming_uniform_(self.cost_trans32a[0].weight, a=1)  # pyre-ignore
            # nn.init.kaiming_uniform_(self.cost_trans32b[0].weight, a=1)  # pyre-ignore
            # nn.init.kaiming_uniform_(self.cost_trans32c[0].weight, a=1)  # pyre-ignore

        self.dres0 = nn.Sequential(ConvBN_3D(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   ConvBN_3D(32, self.hg_size, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(ConvBN_3D(self.hg_size, self.hg_size, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   ConvBN_3D(self.hg_size, self.hg_size, 3, 1, 1))

        self.dres2 = hourglass(self.hg_size)

        self.dres3 = hourglass(self.hg_size)

        self.classif0 = nn.Sequential(ConvBN_3D(self.hg_size, self.hg_size, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.hg_size, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(ConvBN_3D(self.hg_size, self.hg_size, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.hg_size, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(ConvBN_3D(self.hg_size, self.hg_size, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.hg_size, 1, kernel_size=3, padding=1, stride=1, bias=False))

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

    def forward(self, left, right, being_taught=False):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        # print(self.attention_weights.shape)
        # print(volume.shape)
        # volume = self.attention_weights * volume
        cost01 = self.dres0(volume)
        out0 = self.dres1(cost01) + cost01
        #print(cost0.shape)

        out1 = self.dres2(out0)
        out2 = self.dres3(out1)

        if self.training:
            cost0x = self.classif0(out0)
            cost1x = self.classif1(out1)
            cost2x = self.classif2(out2)

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


            if being_taught:

                feat_outputs = []
                soft_outputs = []

                if "early_feats" in self.loss_config['feat_config']:
                    feat_outputs.extend([features_left["early_feats1"], features_left["early_feats2"], \
                                   features_right["early_feats1"], features_right["early_feats2"]])

                if "gwc_vol" in self.loss_config['feat_config']:
                    feat_outputs.extend([gwc_volume])

                if 'hg_vols' in self.loss_config['feat_config']:
                    if self.transform_feats:
                        out0 = self.cost_trans32a(out0)
                        out1 = self.cost_trans32b(out1)
                        out2 = self.cost_trans32c(out2)

                    feat_outputs.extend([out0, out1, out2])

                if 'att_vol' in self.loss_config['feat_config']:
                    att_vol = volume * self.attention_weights
                    feat_outputs.extend([att_vol])

                if 'att_weights' in self.loss_config['feat_config']:
                    feat_outputs.extend([self.attention_weights])

                if 'cost0' in self.loss_config['soft_config']:
                    soft_outputs.extend([cost0])

                if 'cost1' in self.loss_config['soft_config']:
                    soft_outputs.extend([cost1])

                if 'cost2' in self.loss_config['soft_config']:
                    soft_outputs.extend([cost2])


                return [pred0, pred1, pred2], feat_outputs, soft_outputs

            return [pred0, pred1, pred2]

        else:
            cost2 = self.classif2(out2)
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            return [pred2]


def GwcNet_G(d, loss_config="none", transform_feats=False):
    return GwcNet(d, loss_config, transform_feats, use_concat_volume=False)


def GwcNet_GC(d):
    return GwcNet(d, use_concat_volume=True)
