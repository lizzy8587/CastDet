# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
import math
import torch
from copy import deepcopy
from mmcv.ops.diff_iou_rotated import box2corners
from mmdet.models.task_modules.assigners.match_cost import (BaseMatchCost,
                                                            BBoxL1Cost)
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData
from torch import Tensor
from typing import Optional, Union

from mmrotate.models.losses.gaussian_dist_loss import (
    postprocess, xy_stddev_pearson_2_xy_sigma, xy_wh_r_2_xy_sigma)
from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import rbbox_overlaps


def box2multiple_corners(box, num_points):
    B = box.size()[0]
    x, y, w, h, alpha = box.split([1, 1, 1, 1, 1], dim=-1)
    num_segments = num_points // 4

    # Calculate weights for the corners
    weights = torch.linspace(0, 1, num_segments + 1, device=box.device)[:-1]

    # Generate x and y coordinates based on the weights
    x_coords = torch.cat([
        -w / 2 + w * weights, w / 2 * torch.ones_like(weights),
        w / 2 - w * weights, -w / 2 * torch.ones_like(weights)
    ],
                         dim=-1)

    y_coords = torch.cat([
        h / 2 * torch.ones_like(weights), h / 2 - h * weights,
        -h / 2 * torch.ones_like(weights), -h / 2 + h * weights
    ],
                         dim=-1)

    corners = torch.stack([x_coords, y_coords],
                          dim=-1)  # (B, N, num_points, 2)

    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)
    rot_T = torch.stack([row1, row2], dim=-2)

    N = box.size(1)
    rotated = torch.bmm(
        corners.view([-1, num_points, 2]),
        rot_T.repeat(1, 1, 1, 1).view([-1, 2, 2]))
    rotated = rotated.view([B, N, num_points, 2])
    rotated[..., 0] += x
    rotated[..., 1] += y

    return rotated


def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1., sqrt=True):
    """Kullback-Leibler Divergence Edited from 'kld_loss', 'kld_loss' must have
    the Tensors inputs as same shape and have an output as scalar by
    decorator(weighted_loss) for calculating loss.

    Edited function 'kld',
    Calculate kld for each pair for assign.
    Args:
        bbox1 (Tuple[torch.Tensor]): (xy (n,2), sigma(n,2,2))
        bbox2 (Tuple[torch.Tensor]) : (xy (m,2), sigma(m,2,2))
    Returns:
        kullback leibler divergence (torch.Tensor) shape (n, m)
    """
    xy_1, Sigma_1 = pred
    xy_2, Sigma_2 = target

    N, _ = xy_1.shape
    M, _ = xy_2.shape
    xy_1 = xy_1.unsqueeze(1).repeat(1, M, 1).view(-1, 2)
    Sigma_1 = Sigma_1.unsqueeze(1).repeat(1, M, 1, 1).view(-1, 2, 2)
    xy_2 = xy_2.unsqueeze(0).repeat(N, 1, 1).view(-1, 2)
    Sigma_2 = Sigma_2.unsqueeze(0).repeat(N, 1, 1, 1).view(-1, 2, 2)
    return_shape = [N, M]

    Sigma_1_inv = torch.stack((Sigma_1[..., 1, 1], -Sigma_1[..., 0, 1],
                               -Sigma_1[..., 1, 0], Sigma_1[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_1_inv = Sigma_1_inv / Sigma_1.det().unsqueeze(-1).unsqueeze(-1)
    dxy = (xy_1 - xy_2).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_1_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_1_inv.bmm(Sigma_2).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_1_det_log = Sigma_1.det().log()
    Sigma_2_det_log = Sigma_2.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_1_det_log - Sigma_2_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)

    if sqrt:
        distance = distance.clamp(1e-7).sqrt()
    distance = distance.reshape(return_shape)

    return postprocess(distance, fun=fun, tau=tau)


@TASK_UTILS.register_module()
class RBoxL1Cost(BBoxL1Cost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 box_format: str = 'xywha',
                 angle_factor=math.pi,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format == 'xywha'
        self.box_format = box_format
        self.angle_factor = angle_factor

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class CenterL1Cost(RBoxL1Cost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        pred_bboxes = pred_bboxes[..., :2]
        gt_bboxes = gt_bboxes[..., :2]

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class GDCost(BaseMatchCost):
    """IoUCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'giou'. Defaults to 'giou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import IoUCost
        >>> import torch
        >>> self = IoUCost()
        >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
        >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> self(bboxes, gt_bboxes)
        tensor([[-0.1250,  0.1667],
            [ 0.1667, -0.5000]])
    """
    BAG_GD_COST = {
        'gwd': None,
        'kld': kld_loss,
        'jd': None,
        'kld_symmax': None,
        'kld_symmin': None,
    }
    BAG_PREP = {
        'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
        'xy_wh_r': xy_wh_r_2_xy_sigma
    }

    def __init__(self,
                 loss_type,
                 representation='xy_wh_r',
                 fun='log1p',
                 tau=0.0,
                 alpha=1.0,
                 weight: Union[float, int] = 1.,
                 **kwargs):
        super().__init__(weight=weight)
        assert fun in ['log1p', 'none', 'sqrt']
        assert loss_type in self.BAG_GD_COST
        self.cost = self.BAG_GD_COST[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.kwargs = kwargs

    @torch.no_grad()
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        pred_bboxes = self.preprocess(pred_bboxes)
        gt_bboxes = self.preprocess(gt_bboxes)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        return self.cost(
            pred_bboxes,
            gt_bboxes,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            **_kwargs) * self.weight


@TASK_UTILS.register_module()
class HausdorffCost(RBoxL1Cost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self, *args, num_points=4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_points = num_points

    def compute_hausdorff(self, boxes1, boxes2):
        pairwise_distances = torch.norm(
            boxes1[:, None, :, None, :] - boxes2[None, :, None, :, :], dim=-1)
        min_distanecs = pairwise_distances.min(dim=-1)[0]
        max_distances = min_distanecs.max(dim=-1)[0]

        return max_distances

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        assign_on_cpu = False

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h,
                                       1]).unsqueeze(0)
        pred_bboxes = pred_bboxes / factor
        gt_bboxes = gt_bboxes / factor

        if self.num_points == 4:
            corners1 = box2corners(pred_bboxes.unsqueeze(0)).squeeze(0)
            corners2 = box2corners(gt_bboxes.unsqueeze(0)).squeeze(0)
        elif self.num_points > 4:
            corners1 = box2multiple_corners(
                pred_bboxes.unsqueeze(0), self.num_points).squeeze(0)
            corners2 = box2multiple_corners(
                gt_bboxes.unsqueeze(0), self.num_points).squeeze(0)

            if corners2.shape[0] >= 700:
                assign_on_cpu = True
                device = corners2.device
                corners1 = corners1.cpu()
                corners2 = corners2.cpu()
        else:
            raise RuntimeError('Not Adequte Num Points')

        d1 = self.compute_hausdorff(corners1, corners2)
        d2 = self.compute_hausdorff(corners2, corners1).transpose(1, 0)

        if assign_on_cpu:
            d1 = d1.to(device)
            d2 = d2.to(device)

        bbox_cost = torch.maximum(d1, d2)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class RotatedIoUCost(BaseMatchCost):
    """IoUCost.

    Note: ``rboxes`` in ``InstanceData`` passed in is of format 'xywha'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'iof'. Defaults to 'iou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, iou_mode: str = 'iou', weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``rboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, w, h, rad).
            gt_instances (:obj:`InstanceData`): ``rboxes`` inside is gt
                rboxes with unnormalized coordinate (x, y, w, h, rad).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor

        overlaps = rbbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight
