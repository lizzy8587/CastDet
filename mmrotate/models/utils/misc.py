# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import convex_iou
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, InstanceList


def points_center_pts(RPoints, y_first=True):
    """Compute center point of Pointsets.

    Args:
        RPoints (torch.Tensor): the  lists of Pointsets, shape (k, 18).
        y_first (bool, optional): if True, the sequence of Pointsets is (y,x).

    Returns:
        center_pts (torch.Tensor): the mean_center coordination of Pointsets,
            shape (k, 18).
    """
    RPoints = RPoints.reshape(-1, 9, 2)

    if y_first:
        pts_dy = RPoints[:, :, 0::2]
        pts_dx = RPoints[:, :, 1::2]
    else:
        pts_dx = RPoints[:, :, 0::2]
        pts_dy = RPoints[:, :, 1::2]
    pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
    pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
    center_pts = torch.cat([pts_dx_mean, pts_dy_mean], dim=1).reshape(-1, 2)
    return center_pts


def convex_overlaps(gt_bboxes, points):
    """Compute overlaps between polygons and points.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
        points (torch.Tensor): Points to be assigned, shape(n, 18).

    Returns:
        overlaps (torch.Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
    """
    overlaps = convex_iou(points, gt_bboxes)
    overlaps = overlaps.transpose(1, 0)
    return overlaps


def levels_to_images(mlvl_tensor, flatten=False):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
        flatten (bool, optional): if shape of mlvl_tensor is (N, C, H, W)
            set False, if shape of mlvl_tensor is  (N, H, W, C) set True.

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


def get_num_level_anchors_inside(num_level_anchors, inside_flags):
    """Get number of every level anchors inside.

    Args:
        num_level_anchors (List[int]): List of number of every level's anchors.
        inside_flags (torch.Tensor): Flags of all anchors.

    Returns:
        List[int]: List of number of inside anchors.
    """
    split_inside_flags = torch.split(inside_flags, num_level_anchors)
    num_level_anchors_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_anchors_inside

def _filter_gt_instances_by_score(batch_data_samples: SampleList,
                                  score_thr: float) -> SampleList:
    """Filter ground truth (GT) instances by score.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        score_thr (float): The score filter threshold.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        assert 'scores' in data_samples.gt_instances, \
            'there does not exit scores in instances'
        if data_samples.gt_instances.bboxes.shape[0] > 0:
            data_samples.gt_instances = data_samples.gt_instances[
                data_samples.gt_instances.scores > score_thr]
    return batch_data_samples


def _filter_gt_instances_by_size(batch_data_samples: SampleList,
                                 wh_thr: tuple,
                                 bbox_type: str) -> SampleList:
    """Filter ground truth (GT) instances by size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        bboxes = data_samples.gt_instances.bboxes
        if bboxes.shape[0] > 0:
            if bbox_type == 'xyxy':
                w = bboxes[:, 2] - bboxes[:, 0]
                h = bboxes[:, 3] - bboxes[:, 1]
            elif bbox_type == 'xywha':
                w, h = bboxes[:, 2], bboxes[:, 3]
            else:
                raise NotImplementedError
            
            data_samples.gt_instances = data_samples.gt_instances[
                (w > wh_thr[0]) & (h > wh_thr[1])]
    return batch_data_samples

def _filter_gt_instances_by_uncs_score(batch_data_samples: SampleList,
                                  uncs_thr: float) -> SampleList:
    """Filter ground truth (GT) instances by reg uncs score.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        uncs_thr (float): The score filter threshold.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        assert 'reg_uncs' in data_samples.gt_instances, \
            'there does not exit scores in instances'
        if data_samples.gt_instances.bboxes.shape[0] > 0:
            data_samples.gt_instances = data_samples.gt_instances[
                data_samples.gt_instances.reg_uncs < uncs_thr]
    return batch_data_samples

def _filter_rpn_results_by_score(rpn_results_list: InstanceList,
                                  score_thr: float) -> InstanceList:
    """Filter proposals (RPN) instances by score.

    Args:
        rpn_results_list (InstanceList): The Data RPN results.
        score_thr (float): The score filter threshold.

    Returns:
        InstanceList: The RPN resultss filtered by score.
    """
    rpn_results_list = [results[results.scores > score_thr] 
                        for results in rpn_results_list]
    return rpn_results_list

def filter_gt_instances(batch_data_samples: SampleList,
                        score_thr: float = None,
                        wh_thr: tuple = None,
                        uncs_thr: float = None,
                        bbox_type: str = 'xyxy'):
    """Filter ground truth (GT) instances by score and/or size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        score_thr (float): The score filter threshold.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score and/or size.
    """

    if score_thr is not None:
        batch_data_samples = _filter_gt_instances_by_score(
            batch_data_samples, score_thr)
    if uncs_thr is not None:
        batch_data_samples = _filter_gt_instances_by_uncs_score(
            batch_data_samples, uncs_thr)        
    if wh_thr is not None:
        batch_data_samples = _filter_gt_instances_by_size(
            batch_data_samples, wh_thr, bbox_type=bbox_type)
    return batch_data_samples