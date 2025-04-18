dataset_type = 'AgroNavDataset'
data_root = 'data/agronav'

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=[]  # we’ll set this next
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=[]
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',  # or test, if you have it
        ann_dir='annotations/val',
        pipeline=[]
    )
)