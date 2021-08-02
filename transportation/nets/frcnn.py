import torch
import torch.nn as nn
from transportation.nets.vgg16 import decom_vgg16
from transportation.nets.resnet50 import resnet50
from transportation.nets.rpn import RegionProposalNetwork
from transportation.nets.classifier import VGG16RoIHead, Resnet50RoIHead
import time
import numpy as np


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='vgg'
                 ):
        super(FasterRCNN, self).__init__()

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.feat_stride = feat_stride
        if backbone == 'vgg':
            self.extractor, self.classifier = decom_vgg16()
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            self.head = VGG16RoIHead(
                n_class=num_classes + 1,
                roi_size=7,
                spatial_scale=(1. / self.feat_stride),
                classifier=self.classifier
            )
        elif backbone == 'resnet50':
            self.extractor, self.classifier = resnet50()

            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=(1. / self.feat_stride),
                classifier=self.classifier
            )

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x) # 输入一张图片得到其特征图feature map

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn.forward(h, img_size, scale)  # 给定特征图后产生一系列RoIs

        # print(np.shape(h))
        # print(np.shape(rois))
        # print(roi_indices)
        # 利用这些RoIs对应的特征图对这些RoIs中的类别进行分类，并提升定位精度
        roi_cls_locs, roi_scores = self.head.forward(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices