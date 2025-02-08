_base_ = [
    'mmrotate::_base_/models/oriented-rcnn-le90_r50_fpn.py',
    'mmrotate::_base_/default_runtime.py',
    'mmrotate::_base_/datasets/visdronezsd.py'
]

batch_size = 8
num_workers = 2
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,

)

model = dict(
    roi_head  = dict(
        bbox_head = dict(num_classes=16)
    )
)

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor= 1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20000,
        by_epoch=False,
        milestones=[16000, 18000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=5))
log_processor = dict(by_epoch=False)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])

# for debug
# load_from = "work_dirs/soft-teacher_faster-rcnn_r50-caffe_fpn_80k_semi-dior/iter_10000.pth"