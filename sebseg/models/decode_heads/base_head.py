# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

@MODELS.register_module()
class BaselineHead(BaseDecodeHead):
    """Vanilla head for mapping feature to a predefined set
    of classes.

    Args:
        in_channels (int): Number of feature maps coming from 
        the decoded prediction.
            Default: 256.
        num_classes (int): Number of classes in the training
        dataset.
            Default: 19 for Cityscapes.
    """

    def __init__(self, in_channels=256, num_classes=19, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(in_channels, int)
        assert isinstance(num_classes, int)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seg_head = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        """Forward function."""
        output = self.seg_head(x)
        return output