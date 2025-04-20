dataset_type = 'AgroNavDataset'
data_root = 'data/agronav'

classes = ('soil', 'sidewalk', 'vegetation', 'sky', 'human', 'vehicle', 'building', 'wall', 'others')
palette = [[128, 64, 128], [244, 35, 232], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142], [70, 70, 70], [102, 102, 156], [0, 0, 0]]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')  # Required in MMSeg 2.x
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline,
        classes=classes,
        palette=palette
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=val_pipeline,
        classes=classes,
        palette=palette
    )
)

test_dataloader = val_dataloader  # If you want to use val settings for test

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
