import torch.nn as nn

from ..builder import HEADS
from ..modules import CASENet
from mmengine.model import BaseModule


@HEADS.register_module()
class CASENetHead(BaseModule):
    pass
