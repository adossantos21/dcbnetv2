'''
Auxiliary modules for semantic segmentation models.
For PIDNet, we have the P Branch and the D Branch modules.
For Semantic Boundary Detection (SBD), we have the CASENet, DFF, and BGF modules.
'''

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from mmseg.models.utils import BasicBlock, Bottleneck
from .base import CustomBaseModule
from .fusion_modules import (
    PagFM,
    Bag,
    LightBag,
    PIFusion,
)


class PModule(CustomBaseModule):
    '''
    Model layers for the P branch of PIDNet. 

    Args:
        in_channels (int): The number of input channels. Default: 3.
        channels (int): The number of channels in the stem layer. Default: 64.
        ppm_channels (int): The number of channels in the PPM layer.
            Default: 96.
        num_stem_blocks (int): The number of blocks in the stem layer.
            Default: 2.
        num_branch_blocks (int): The number of blocks in the branch layer.
            Default: 3.
        align_corners (bool): The align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    '''

    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.align_corners = align_corners

        self.relu = nn.ReLU()

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
        x_2, x_3, x_4 = x

        # stage 3
        x_p = self.p_branch_layers[0](x_2)

        comp_i = self.compression_1(x_3)
        x_p = self.pag_1(x_p, comp_i)
        if self.training:
            temp_p = x_p.clone()

        # stage 4
        x_p = self.p_branch_layers[1](self.relu(x_p))

        comp_i = self.compression_2(x_4)
        x_p = self.pag_2(x_p, comp_i)

        # stage 5
        x_p = self.p_branch_layers[2](self.relu(x_p))
        
        if self.training: 
            return temp_p, x_p

class DModule(CustomBaseModule):
    '''
    Model layers for the D branch of PIDNet.
    '''
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        self.relu = nn.ReLU()

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
        x_2, x_3, x_4 = x

        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8


        # stage 3
        x_d = self.d_branch_layers[0](x_2)

        diff_i = self.diff_1(x_3)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 4
        x_d = self.d_branch_layers[1](self.relu(x_d))

        diff_i = self.diff_2(x_4)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_d = x_d.clone()

        # stage 5
        x_d = self.d_branch_layers[2](self.relu(x_d))

        if self.training:
            return temp_d, x_d

class CASENet(BaseModule):
    '''
    Model layers for the CASENet SBD module.
    '''
    pass
    
class DFF(BaseModule):
    '''
    Model layers for the Dynamic Feature Fusion (DFF) SBD module.
    '''
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DFF, self).__init__(nclass, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass
        self.ada_learner = LocationAdaptiveLearner(nclass, nclass*4, nclass*4, norm_layer=norm_layer)
        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1)) # finish coding this

    def forward(self, x):
        c1, c2, c3, _, c5, _ = x # finish coding this too

class BGF(BaseModule):
    '''
    Model layers for DCBNetv1's SBD module.
    '''
    pass