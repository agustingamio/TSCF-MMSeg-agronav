_base_ = [
    '../san/san-vit-l14_coco-stuff164k-640x640.py',
]

find_unused_parameters = True

# Freeze the image and text encoders by setting lr_mult=0.0
paramwise_cfg = dict(
    custom_keys={
        'image_encoder': dict(lr_mult=0.0),
        'text_encoder': dict(lr_mult=0.0)
    }
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=paramwise_cfg
)
