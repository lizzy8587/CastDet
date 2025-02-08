_base_ = [
    'mmrotate::_base_/models/oriented-rcnn-le90_r50_fpn.py',
    'mmrotate::_base_/default_runtime.py',
    'mmrotate::_base_/datasets/semi_visdronezsd.py'
]
work_dir = 'work_dirs/visdrone_step2_castdet_12b_10k_oriented'

batch_size = 8
num_workers = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(
        batch_size=batch_size,
        source_ratio=[1, 3]
    )
)

# visdrone_zsd split
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers
)

# dior split
test_dataloader = dict(
    dataset=dict(
        ann_file='ImageSets/Main/test.txt',
    )
)

custom_imports = dict(
    imports=['projects.CastDetv2.castdet'], allow_failed_imports=False)


detector = _base_.model
detector.data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[122.7709383 , 116.7460125 , 104.09373615],
    std=[68.5005327 , 66.6321579 , 70.32316305],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    boxtype2tensor=False)
detector.roi_head.bbox_head.type='Shared2FCBBoxHeadZSD'
detector.roi_head.bbox_head.num_classes=20
detector.roi_head.bbox_head.reg_class_agnostic=True
detector.roi_head.bbox_head.fc_cls=dict(
                    type='Projection2',
                    vector_path="projects/CastDetv2/resources/remoteCLIP_embeddings_normalized.npy",
                    is_scale=True,
                    is_grad_bg=True,
                    is_grad=False
                )

model = dict(
    _delete_=True,
    type='RotatedCastDet',
    detector=detector,
    bbox_type='xywha',
    rpn_bbox_type='xywha',
    visual=dict(
        type='ModifiedResNet2',
        layers=[3, 4, 6, 3],
        width=64,
        output_dim=1024,
        heads=32,
        image_size=224,
    ),
    pseudo_queue_cfg=dict(
        type='PseudoQueue',
        unseen_ids=(16, 17, 18, 19),
        sample_prob=(0.4, 0.1, 0.1, 0.4),
        batch_num=4,
        start_train_num=3,
        start_train_iter=2000,
        initialize=True,
        save_path=f"{work_dir}/save_queue_samples.npz",
        init_imgs_path="projects/CastDetv2/resources/visdronezsd_split/visdrone_initialize.txt"
    ),
    data_preprocessor=dict(
        type='mmdet.MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        pseudo_label_initial_score_thr=0.5, # seems not used?
        rpn_pseudo_thr=0.9, # seems not used?
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        jitter_angle_scale=0.1,
        min_pseudo_bbox_wh=(1e-2, 1e-2),
        semi_weight=1.0,    # semi branch
        semi_reg_iter=10,
        semi_jitter_uncs_thr=0.03,
        semi_jitter_angle_thr=0.05,
        sin_norm=True,
        semi_max_rpn_num=16,
        semi_min_rpn_num=0,
        semi_rpn_score=0.3,
        semi_cls_score=0.95,
        semi_min_size=1000,
        semi_min_label=16,
        semi_bbox_loss=True,
        ignore_bg=False,
        semi_bg_weight=0.5,
        unsup_cls_loss=True,
        start_unsup_iter=0,
        initial_rpn_score_thr=0.3,
        vector_path="projects/CastDetv2/resources/remoteCLIP_embeddings_bgs_normalized.npy"
    ),
    semi_test_cfg=dict(predict_on='teacher'))


# training schedule for 10k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=10000, val_interval=2000)
val_cfg = dict(type='mmdet.TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'visual': dict(decay_mult=0.)
        },
        norm_decay_mult=0.))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=1))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook')]

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])

load_from = 'work_dirs/oriented-rcnn_r50-fpn_20k_visdronezsd_base-set/merged_castdet_init_iter20k.pth'
