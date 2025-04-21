_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/agronav_dataset.py',
    '../_base_/schedules/schedule_20k.py',
    '../_base_/default_runtime.py'
]

# Adjust number of classes
num_classes = 9  # soil, sidewalk, vegetation, ...

model = dict(
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes)
)

# Dataset root
data_root = '/content/TSCF-MMSeg-agronav/data/agronav'

# Paths are set in the dataset config but can be overridden here
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='AgroNavDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        pipeline=None  # uses default pipeline from base dataset config
    )
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='AgroNavDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=None
    )
)

# Evaluation and logging
val_evaluator = dict(type='IoUMetric', metric='mIoU')
test_evaluator = val_evaluator

# Logging frequency
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50)
)

# Training schedule (optional tweaks)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=16000, val_interval=1600)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)
