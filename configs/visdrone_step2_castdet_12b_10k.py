_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py', 'mmdet::_base_/default_runtime.py',
    './semi_visdrone_detection.py'
]
work_dir = 'work_dirs/visdrone_step2_castdet_12b_10k'

custom_imports = dict(
    imports=['castdet'], allow_failed_imports=False)

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[122.7709383 , 116.7460125 , 104.09373615],
    std=[68.5005327 , 66.6321579 , 70.32316305],
    bgr_to_rgb=True,
    pad_size_divisor=32)
detector.backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

detector.roi_head=dict(
    type='StandardRoIHead',
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(
            type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='Shared2FCBBoxHeadZSD',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=20,
        reg_class_agnostic=True,
        fc_cls=dict(
                type='Projection2',
                vector_path="resources/remoteCLIP_embeddings_normalized.npy",
                is_scale=True,
                is_grad_bg=True,
                is_grad=False
            ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))

detector.test_cfg=dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

model = dict(
    _delete_=True,
    type='CastDet',
    detector=detector,
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
        start_train_num=30,
        start_train_iter=0,
        initialize=False,
        save_path=f"{work_dir}/save_queue_samples.npz",
        init_imgs_path=None
    ),
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2),
        semi_weight=1.0,    # semi branch
        semi_max_rpn_num=16,
        semi_min_rpn_num=3,
        semi_rpn_score=0.95,
        semi_cls_score=0.8,
        semi_min_size=1000,
        semi_min_label=16,
        semi_bbox_loss=False,
        ignore_bg=False,
        semi_bg_weight=0.05,
        unsup_cls_loss=True,
        vector_path="resources/remoteCLIP_embeddings_bgs_normalized.npy"
    ),
    semi_test_cfg=dict(predict_on='teacher'))


# training schedule for 10k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
val_cfg = dict(type='TeacherStudentValLoop')
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

load_from = 'checkpoints/init_80k.pth'