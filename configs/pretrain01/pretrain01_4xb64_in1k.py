# Default Pre-training Config

_base_ = [
    '../_base_/models/basenet.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
