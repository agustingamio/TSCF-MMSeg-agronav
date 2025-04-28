_base_ = [
    '../_base_/models/mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024.py',
    '../_base_/datasets/agronav_dataset.py',
    '../_base_/schedules/schedule_20k.py',
    '../_base_/default_runtime.py'
]

# Number of classes
num_classes = 9

# Use Group Normalization
norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)

model = dict(
    data_preprocessor=dict(size=(512, 512)),
    backbone=dict(norm_cfg=norm_cfg),  # override backbone normalization
    decode_head=dict(
        num_classes=num_classes,
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        norm_cfg=norm_cfg
    )
)

# Dataset root
data_root = '/content/TSCF-MMSeg-agronav/data/agronav'

# Corrected train and val dataloaders (pipeline removed completely)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='AgroNavDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train')
    )
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='AgroNavDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val')
    )
)

# Evaluation and logging
val_evaluator = dict(type='IoUMetric', metric='mIoU')
test_evaluator = val_evaluator

# Logging frequency
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50)
)

# Training loop
train_cfg = dict(type='IterBasedTrainLoop', max_iters=16000, val_interval=1600)

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)
