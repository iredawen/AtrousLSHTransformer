_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='PVT_AtrousLSHTransformer',
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='/home/dl4/x/AtrousLSHTransformer/'
                     'checkpoints/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))

    
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12) 