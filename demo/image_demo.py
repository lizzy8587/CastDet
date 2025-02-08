# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector   #, init_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules


import copy
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import torch.nn as nn
from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet.registry import DATASETS
from mmdet.evaluation import get_classes
from mmrotate.registry import MODELS

def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'none',
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif hasattr(config.model, 'backbone') and 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    elif hasattr(config.model, 'detector') and 'init_cfg' in config.model.detector.backbone:
        config.model.detector.backbone.init_cfg = None
    init_default_scope(config.get('default_scope', 'mmdet'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter('once')
        warnings.warn('checkpoint is None, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})

        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

    # Priority:  args.palette -> config -> checkpoint
    if palette != 'none':
        model.dataset_meta['palette'] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car ."')
    # only for GLIP and Grounding DINO
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    call_args = dict(
        model=model,
        imgs=args.img,
        text_prompt=args.texts,
        custom_entities=args.custom_entities,
    )
    result = inference_detector(**call_args)

    # show the results
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=True,
        show=args.out_file is None,
        wait_time=0,
        out_file=args.out_file,
        pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
