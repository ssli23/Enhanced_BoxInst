# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import argparse
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class cross_update_block(nn.Module):
    def __init__(self, n_class):
        super(cross_update_block, self).__init__()
        self.n_class = n_class
        self.softmax = Softmax(dim=-1)

    def forward(self, refined_shape_prior, feature):
        class_feature = torch.matmul(feature.flatten(2), refined_shape_prior.flatten(2).transpose(-1, -2))
        # scale
        class_feature = class_feature / math.sqrt(self.n_class)
        class_feature = self.softmax(class_feature)

        class_feature = torch.einsum("ijk, ikhw->ijhw", class_feature, refined_shape_prior)
        class_feature = feature + class_feature
        return class_feature

class self_attention(nn.Module):
    def __init__(self, in_dim):
        super(self_attention,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels= self.chanel_in, out_channels= self.chanel_in // 8,kernel_size =1)
        self.key_conv = nn.Conv2d(in_channels= self.chanel_in, out_channels= self.chanel_in // 8,kernel_size =1)
        self.value_conv = nn.Conv2d(in_channels= self.chanel_in, out_channels= self.chanel_in ,kernel_size =1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim = -1)

        # self.position_embeddings = nn.Parameter(
        #     torch.randn(1, self.num_attention_heads, n_classes, n_classes))

    def forward(self,x, x2):
        """
        inputs:bcwh
        return: out :sele attention value + input feature
                attention:BxNxN(N = WxH)
        """
        m_batchsize , C, width,height = x.size()
        proj_query = self.query_conv(x2).view(m_batchsize, -1, width*height).permute(0,2,1) #BxCxN
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height) # BxCx(w*h)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy)  ##BxNxN
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        # return out, attention
        return out


class SPM(nn.Module):
    def __init__(self, n_classes, in_channel):
        super(SPM, self).__init__()
        self.attention = self_attention(in_dim = in_channel)
        self.CUB = cross_update_block(n_classes)
        self.resblock1 = DecoderResBlock(in_channel, in_channel)
        self.resblock2 = DecoderResBlock(in_channel, in_channel)
        self.resblock3 = DecoderResBlock(in_channel, in_channel)

        self.dim = in_channel

    def forward(self, feature, refined_shape_prior):
        # print(refined_shape_prior.size())
        b, _, _, _ = refined_shape_prior.size()
        refined_shape_prior = self.attention(refined_shape_prior,feature)
        previous_class_center = refined_shape_prior

        feature = self.resblock1(feature)
        feature = self.resblock2(feature)

        class_feature = self.CUB(refined_shape_prior, feature)

        # b * N * H/i * W/i * L/i
        refined_shape_prior = self.resblock3(class_feature)

        refined_shape_prior = refined_shape_prior + previous_class_center

        return class_feature, refined_shape_prior


class Conv2dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dbn, self).__init__(conv, bn)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv3 = Conv2dbn(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        feature_in = self.conv3(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + feature_in
        x = self.relu(x)
        # x = self.se_block(x)

        return x





n_classes = 1
# 生成随机张量
tensor1 = torch.rand(1, 256, 25, 25)
tensor2 = torch.rand(1, 256, 50, 50)
tensor3 = torch.rand(1, 256, 100, 100)
# print(tensor)
# 将张量保存在 GPU 中
tensor1 = tensor1.to('cuda')
tensor2 = tensor2.to('cuda')
tensor3 = tensor3.to('cuda')

spm1 = SPM(n_classes = n_classes,  in_channel = 256).to('cuda')
spm2 = SPM(n_classes = n_classes,  in_channel = 256).to('cuda')
spm3 = SPM(n_classes = n_classes,  in_channel = 256).to('cuda')
learnable_shape_prior = nn.Parameter(torch.randn(1, 256,50, 50))
B = tensor1.size()[0]
learnable_shape_prior = learnable_shape_prior.repeat(B, 1, 1, 1).to('cuda')
# out1 = spm1(tensor1,learnable_shape_prior)
out2 = spm2(tensor2,learnable_shape_prior)
# out3 = spm3(tensor3,learnable_shape_prior)
# print(out1[0].shape)
# print(out1[1].shape)
print(out2[0].shape)
print(out2[1].shape)
# print(out3[0].shape)
# print(out3[1].shape)
