_base_ = [
    '../_base_/datasets/agronav_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# Model configuration
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)

# Model definition - Fully explicit to ensure compatibility with checkpoint
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MobileNetV3',
        arch='large',
        out_indices=(1, 3, 16),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 24, 960),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=9,  # Explicitly set number of classes
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Add palette for visualization
model['decode_head']['palette'] = [
    [128, 64, 128],    # background - blue
    [0, 255, 0],       # crop - green
    [255, 0, 0],       # weed - red
    [165, 42, 42],     # soil - brown
    [0, 128, 0],       # grass - dark green
    [255, 165, 0],     # obstacle - orange
    [255, 192, 203],   # person - pink
    [135, 206, 235],   # sky - sky blue
    [0, 0, 0]          # ignore - black
]

# Runtime settings
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Make sure we're using the new dataset implementation
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='AgroNavDataset',
        data_root='data/agronav',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='annotations/train',
            split='splits/train.txt'
        ),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AgroNavDataset',
        data_root='data/agronav',
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val',
            split='splits/val.txt'
        ),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Logging and checkpoint configuration
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None  # Set to your checkpoint path when testing
resume = False

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

# Fixed values
seed = 0
gpu_ids = range(1)
work_dir = './work_dirs/agronav_mobilenet_v3'
device = 'cuda'