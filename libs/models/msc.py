#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits = self.base(x)
        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        print (f'self.scales : {self.scales}')
        for p in self.scales:
            #图片升采或降采
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            print(f'p : {p},{h.shape}')
            logits_pyramid.append(self.base(h))
        print (f'logits_pyramid: 0.5-{logits_pyramid[0].shape},0.75-{logits_pyramid[1].shape}')
        # Pixel-wise max。logit值按原尺寸升采或降采。logit_max即原尺寸
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        print (f'msc_logits_all_len : {len(logits_all)}, logit-- {logits_all[0].shape},{logits_all[1].shape},{logits_all[2].shape},') # 100%,50%,75%
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0] #dim=0每列索引最大值

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max
