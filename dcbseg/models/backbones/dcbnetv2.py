# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from mmseg.models.utils import DAPPM, PAPPM, BasicBlock, Bottleneck
from .base import BackboneBaseModule


@MODELS.register_module()
class DCBNetv2(BackboneBaseModule):
    """DCBNetv2 backbone.

    This backbone is the implementation of `DCBNetv2: Real-Time Semantic
    Segmentation with Semantic Boundary Detection Conditioning.

    Licensed under the MIT License.

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
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer - we need better granularity to integrate the SBD modules
        self.conv1 =  nn.Sequential([
             ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ])
        self.stage_1 = self._make_layer(
            block=BasicBlock,
            in_channels=channels,
            channels=channels,
            num_blocks=num_stem_blocks)
        self.stage_2 = self._make_layer(
            block=BasicBlock,
            in_channels=channels,
            channels=channels * 2,
            num_blocks=num_stem_blocks,
            stride=2)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))

        # Spatial Pyramid Pooling and Fusion Modules
        if num_stem_blocks == 2:
            spp_module = PAPPM
        else:
            spp_module = DAPPM

        self.spp = spp_module(
            channels * 16, ppm_channels, channels * 4, num_scales=5)

    def forward(self, config, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0
        x = self.conv1(x)

        # stage 1
        x_1 = self.relu(self.stage_1(x))

        # stage 2
        x_2 = self.relu(self.stage_2(x_1))

        # stage 3
        x_3 = self.relu(self.i_branch_layers[0](x_2))

        # stage 4
        x_4 = self.relu(self.i_branch_layers[1](x_3))

        # stage 5
        x_5 = self.i_branch_layers[2](x_4)

        x_spp = self.spp(x_5) # performs adaptive avg pooling at several scaled kernels: {5, 9, 17}
        x_out = F.interpolate(
            x_spp,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if config.ABLATION == 0:
            return x_out
        elif config.ABLATION == 1 or config.ABLATION == 2 or config.ABLATION == 4: # PI Model (1), ID Model (2), PID Model (4)
            return (x_2, x_3, x_4, x_out)
        else: # I SBD Model (3), PI SBD Model (5), ID SBD Model (6), PID SBD Model (7)
            return (x_1, x_2, x_3, x_4, x_5, x_out)


