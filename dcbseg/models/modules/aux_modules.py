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

class PModule(BaseModule):
    '''
    Model layers for the P branch of PIDNet. 
    '''
    pass

class DModule(BaseModule):
    '''
    Model layers for the D branch of PIDNet.
    '''
    pass

class CASENet(BaseModule):
    '''
    Model layers for the CASENet SBD module.
    '''
    pass
    
class DFF(BaseModule):
    '''
    Model layers for the Dynamic Feature Fusion (DFF) SBD module.
    '''
    pass

class BGF(BaseModule):
    '''
    Model layers for DCBNetv1's SBD module.
    '''
    pass