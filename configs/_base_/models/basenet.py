# model settings, no neck because spatial pyramid pooling occurs in the backbone
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEBNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
)
