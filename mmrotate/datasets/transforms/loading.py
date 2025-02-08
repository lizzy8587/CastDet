# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union
import torch

import mmcv
from mmcv.transforms import BaseTransform

from mmrotate.registry import TRANSFORMS
from mmdet.datasets.transforms import LoadEmptyAnnotations as LoadEmptyHbbAnnotations
import numpy as np
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks
from mmdet.structures.bbox import get_box_type

@TRANSFORMS.register_module()
class LoadEmptyAnnotations(LoadEmptyHbbAnnotations):
    def __init__(self,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_mask: bool = False,
                 with_seg: bool = False,
                 seg_ignore_label: int = 255,
                 box_type: str = 'rbox') -> None:
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.seg_ignore_label = seg_ignore_label
        self.box_type = box_type

    def transform(self, results: dict) -> dict:
        """Transform function to load empty annotations.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Updated result dict.
        """

        if self.with_bbox:
            results['gt_bboxes'] = np.zeros((0, 5), dtype=np.float32)

            if self.box_type is None:
                results['gt_bboxes'] = np.zeros((0, 5), dtype=np.float32)
            else:
                _, box_type_cls = get_box_type(self.box_type)
                results['gt_bboxes'] = box_type_cls([], dtype=torch.float32)

            results['gt_ignore_flags'] = np.zeros((0, ), dtype=bool)
        if self.with_label:
            results['gt_bboxes_labels'] = np.zeros((0, ), dtype=np.int64)
        if self.with_mask:
            # TODO: support PolygonMasks
            h, w = results['img_shape']
            gt_masks = np.zeros((0, h, w), dtype=np.uint8)
            results['gt_masks'] = BitmapMasks(gt_masks, h, w)
        if self.with_seg:
            h, w = results['img_shape']
            results['gt_seg_map'] = self.seg_ignore_label * np.ones(
                (h, w), dtype=np.uint8)
        return results

@TRANSFORMS.register_module()
class LoadPatchFromNDArray(BaseTransform):
    """Load a patch from the huge image w.r.t ``results['patch']``.

    Requaired Keys:

    - img
    - patch

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        pad_val (float or Sequence[float]): Values to be filled in padding
            areas. Defaults to 0.
    """

    def __init__(self,
                 pad_val: Union[float, Sequence[float]] = 0,
                 **kwargs) -> None:
        self.pad_val = pad_val

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with image array in ``results['img']``
                and patch position in ``results['patch']``.

        Returns:
            dict: The dict contains loaded patch and meta information.
        """
        image = results['img']
        img_h, img_w = image.shape[:2]

        patch_xmin, patch_ymin, patch_xmax, patch_ymax = results['patch']
        assert (patch_xmin < img_w) and (patch_xmax >= 0) and \
            (patch_ymin < img_h) and (patch_ymax >= 0)
        x1 = max(patch_xmin, 0)
        y1 = max(patch_ymin, 0)
        x2 = min(patch_xmax, img_w)
        y2 = min(patch_ymax, img_h)
        padding = (x1 - patch_xmin, y1 - patch_ymin, patch_xmax - x2,
                   patch_ymax - y2)

        patch = image[y1:y2, x1:x2]
        if any(padding):
            patch = mmcv.impad(patch, padding=padding, pad_val=self.pad_val)

        results['img_path'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape[:2]
        results['ori_shape'] = patch.shape[:2]
        return results
