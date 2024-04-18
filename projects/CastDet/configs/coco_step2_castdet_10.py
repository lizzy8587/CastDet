# follow the setting of: Detecting Twenty-thousand Classes using Image-level Supervision
_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py', 'mmdet::_base_/default_runtime.py',
    'mmdet::_base_/datasets/semi_coco_detection.py'
]

work_dir = 'work_dirs/coco_step2_castdet_10'

custom_imports = dict(
    imports=['projects.CastDet.castdet'], allow_failed_imports=False)

frozen_stages = ['visual', 'backbone']

batch_size=12
num_workers=2

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
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='open-mmlab://detectron2/resnet50_caffe'))

detector.roi_head.bbox_head=dict(
        type='Shared2FCBBoxHeadZSD',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=65,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=True,
        fc_cls=dict(
                type='Projection2',   # DictAttentionProjection | Projection | DictFeatureEnhanceProjection
                vector_path="/hhd/datasets/ly/coco2014/word_embeddings/clipRN50_coco65_bg_embeddings_normalized.npy",
                is_scale=True,
                is_grad_bg=True,
                is_grad=False
            ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0))

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
        unseen_ids=(4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63),
        sample_prob=None,
        batch_num=4,
        start_train_num=1,
        start_train_iter=2000,
        transform='coco_transforms',
        initialize=True,
        save_path=f"{work_dir}/save_queue_samples.npz",
        init_imgs_path="/hhd/datasets/ly/ImageNetSubset/initial_samples.txt",
    ),
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2),
        semi_weight=1.0,    # semi branch
        start_semi_iter=2000,   # train student only
        semi_max_rpn_num=64,
        semi_min_rpn_num=16,
        pseudo_nms=True,
        max_keep=16,
        semi_rpn_score=0.95,
        semi_cls_score=0.95,
        semi_min_size=1000,
        semi_crop_ratio=0.1,
        semi_reg_iter=10,
        semi_crop_square=True,
        semi_min_label=(4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63, 65),
        semi_bbox_loss=False,
        ignore_bg=False,
        semi_bg_weight=0.05,
        unsup_cls_loss=True,
        vector_path="/hhd/datasets/ly/coco2014/word_embeddings/clipRN50_coco65_bg_embeddings_normalized.npy",
        clip_logit_scale=85.3745,   # logit_scale.exp()        
        ),
    semi_test_cfg=dict(predict_on='teacher'))


# dataset
from configs._base_.datasets.data_classes import coco65_classes as classes
metainfo = {
    'classes': classes
}
dataset_type = 'CocoDataset'
data_root = '/hhd/datasets/ly/coco2014/'
labeled_dataset = _base_.labeled_dataset
# labeled_dataset.ann_file = 'annotations/instances_val2017_.json'  # for debug
labeled_dataset.ann_file = 'annotations/instances_train2017_seen.json'
labeled_dataset.data_prefix = dict(img='train2014/')
labeled_dataset.data_root = data_root
labeled_dataset.metainfo = metainfo
unlabeled_dataset = _base_.unlabeled_dataset
# unlabeled_dataset.ann_file = 'annotations/instances_val2017_.json'
unlabeled_dataset.ann_file = 'annotations/instances_train2017_seen_50_unseen.json'
unlabeled_dataset.data_prefix = dict(img='train2014/')
unlabeled_dataset.data_root = data_root
unlabeled_dataset.metainfo = metainfo


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(batch_size=batch_size,
                 source_ratio=[1, 1]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo = metainfo,
        ann_file='annotations/instances_val2017_.json',
        data_prefix=dict(img='train2014/')))
# test_dataloader = val_dataloader
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_.json',
                     classwise=True,)
test_evaluator = val_evaluator

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=30000, val_interval=1000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=50))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook')]

# load_from = "/hhd/datasets/ly/SAVE_FILES/mmdetection/work_dirs/coco_step1_faster-rcnn/iter_24216_visual.pth"
load_from = "/hhd/datasets/ly/SAVE_FILES/mmdetection/work_dirs/coco_step1_mask-rcnn_full_seen_/epoch_12_castdet_init.pth"
# fp16=None