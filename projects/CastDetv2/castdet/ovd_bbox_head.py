#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   ovd_bbox_head.py
@Version      :   1.0
@Time         :   2024/04/08 21:05:17
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   open-vocabulary detection head.
'''

from typing import Optional, Tuple, Union
from mmengine.config import ConfigDict
from torch import Tensor
from mmdet.models import ConvFCBBoxHead
from mmrotate.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@MODELS.register_module()
class Projection2(nn.Module):
    """
    Projection layer for ZSD,
    allow the background embedding learnable
    """

    def __init__(self, vector_path, feature_dim, *args, **kwargs):
        super(Projection2, self).__init__()
        is_grad = kwargs.get('is_grad', False)
        is_grad_bg = kwargs.get('is_grad_bg', True)
        
        self.words = nn.Parameter(torch.tensor(np.load(vector_path)[:-1]), requires_grad=is_grad)
        self.bg = nn.Parameter(torch.tensor(np.load(vector_path))[-1][None,:], requires_grad=is_grad_bg)

        self.fc_proj = nn.Linear(feature_dim, self.words.shape[1])
        
        self.is_scale = kwargs.get('is_scale', False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True) if self.is_scale else None

        mask_ids = kwargs.get('train_mask_ids', None)
        self.train_mask = None
        if mask_ids is not None:
            self.train_mask = torch.ones(self.words.shape[0] + self.bg.shape[0], dtype=bool).to(self.words.device)
            self.train_mask[mask_ids] = False
            
    def get_words(self, with_bg=True):
        if with_bg:
            return torch.cat([self.words, self.bg], dim=0)
        else:
            return self.words

    def forward(self, feature, only_feature=False, only_logit=False):
        assert not only_feature or not only_logit, 'Identity'
        
        proj_feature = self.fc_proj(feature)
        
        if only_feature:
            return proj_feature
        
        if self.is_scale:
            probs0 = torch.einsum('bd,cd->bc', F.normalize(proj_feature, dim=-1), F.normalize(self.words, dim=-1))
            probs1 = torch.einsum('bd,cd->bc', F.normalize(proj_feature, dim=-1), F.normalize(self.bg, dim=-1))
            probs = self.logit_scale.exp() * torch.cat([probs0, probs1], dim=-1)
        else:
            probs0 = torch.einsum('bd,cd->bc', proj_feature, self.words)
            probs1 = torch.einsum('bd,cd->bc', proj_feature, self.bg)
            probs = torch.cat([probs0, probs1], dim=-1)
        
        if self.train_mask is not None and self.training:
            probs = probs[:,self.train_mask]
        
        return probs



@MODELS.register_module()
class Shared2FCBBoxHeadZSD(ConvFCBBoxHead):

    def __init__(self,
                 fc_out_channels: int = 1024,
                 fc_cls: Optional[Union[dict, ConfigDict]] = None,
                 *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        if fc_cls is not None:
            fc_cls.update(feature_dim=fc_out_channels)
            self.fc_cls = MODELS.build(fc_cls)

    def forward_reg(self, x: Tuple[Tensor]) -> tuple:
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_reg = x

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return bbox_pred
