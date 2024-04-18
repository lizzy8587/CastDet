_base_ = [
        'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
        'mmdet::_base_/datasets/coco_instance.py',
        'mmdet::_base_/schedules/schedule_2x.py',
        'mmdet::_base_/default_runtime.py'
    ]
dataset_type = 'CocoDataset'
from utils.data_classes import coco65_seen48 as classes
batch_size = 16
# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    data_preprocessor = dict(
        type='DetDataPreprocessor',
        mean=[122.7709383 , 116.7460125 , 104.09373615],
        std=[68.5005327 , 66.6321579 , 70.32316305],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone = dict(
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
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=48,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
        ),
        mask_head=dict(class_agnostic=True),
    ))

# 修改数据集相关配置
data_root = '/hhd/datasets/ly/coco2014/'
metainfo = {
    'classes': classes
}
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017_seen_full.json',
        data_prefix=dict(img="train2014/")))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017_.json',
        data_prefix=dict(img="train2014/")))
test_dataloader = val_dataloader


val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_.json',
                     classwise=True,)
test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=30))
log_processor = dict(by_epoch=True)

# auto_scale_lr = dict(enable=True)



