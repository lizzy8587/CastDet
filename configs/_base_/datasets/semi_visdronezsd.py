# dataset settings
dataset_type = 'DIORDataset'
data_root = 'data/DIOR/'
backend_args = None
batch_size = 5
num_workers = 5

color_space = [
    [dict(type='mmdet.ColorTransform')],
    [dict(type='mmdet.AutoContrast')],
    [dict(type='mmdet.Color')],
    [dict(type='mmdet.Contrast')],
    [dict(type='mmdet.Brightness')],
]

scale = [(800, 800), (1024, 1024)]

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(type='mmdet.Pad', size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        sup=dict(type='mmdet.PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    # dict(type='mmdet.Pad', size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='mmdet.RandAugment', aug_space=color_space, aug_num=1),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180),
    # dict(type='mmdet.Pad', size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadEmptyAnnotations'),
    # dict(type='mmdet.RandomResize', scale=scale, keep_ratio=True),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='ImageSets/Main/visdrone_labeled_3000.txt',
    data_prefix=dict(img_path='JPEGImages-trainval'),
    ann_subdir='Annotations/Oriented Bounding Boxes',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=backend_args)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='ImageSets/Main/visdrone_unlabeled_8726.txt',
    data_prefix=dict(img_path='JPEGImages-trainval'),
    ann_subdir='Annotations/Oriented Bounding Boxes',
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

val_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/visdrone_test.txt',
        data_prefix=dict(img_path='JPEGImages-test'),
        ann_subdir='Annotations/Oriented Bounding Boxes',
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 4]
        ),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
