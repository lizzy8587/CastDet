# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
import torch
from mmdet.models.task_modules.assigners import HungarianAssigner
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from typing import Optional

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DNGroupHungarianAssigner(HungarianAssigner):
    INF = 100.0

    def assign(self,
               pred_instances: InstanceData,
               dn_instances: InstanceData,
               gt_instances: InstanceData,
               dn_meta: dict,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds, num_dn_preds = len(gt_instances), len(
            pred_instances), len(dn_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_dn_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_dn_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        num_groups = dn_meta['num_denoising_groups']

        # 2. compute weighted cost
        cost_list = []
        dn_cost_list = []
        for match_cost in self.match_costs:
            # num_pred x num_gt
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)

            # group x num_gt(pos_dn_query) x num_gt
            dn_cost = match_cost(
                pred_instances=dn_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            dn_cost_list.append(dn_cost)

        # 1. re assign -1 by default
        temp_assigned_gt_inds = torch.full(
            (num_groups * (num_gts + num_preds), ),
            -1,
            dtype=torch.long,
            device=device)
        temp_assigned_labels = torch.full(
            (num_groups * (num_gts + num_preds), ),
            -1,
            dtype=torch.long,
            device=device)
        dn_cost = torch.stack(dn_cost_list).sum(dim=0).view(
            num_groups, num_gts, num_gts)
        cost = torch.stack(cost_list).sum(dim=0)

        # num_groups x (pos_dn_query+num_pred) x num_gt
        all_costs = torch.cat([dn_cost, cost[None].repeat(num_groups, 1, 1)],
                              dim=1)

        all_costs_diag = torch.block_diag(*all_costs)
        ignore_region = ~torch.block_diag(
            *torch.ones_like(all_costs)).bool() * self.INF

        new_cost = all_costs_diag + ignore_region

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        new_cost = new_cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(new_cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            device) % num_gts

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        temp_assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        temp_assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        temp_assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        # 5. extract pos_dn_idx
        pos_range = torch.arange(num_gts, dtype=torch.long, device=device)
        pos_ind = pos_range.repeat(num_groups)

        group_range = torch.arange(num_groups, dtype=torch.long, device=device)
        ind_offset = group_range[..., None].repeat(1, num_gts).view(-1) * (
            num_preds + num_gts)
        pos_dn_ind = pos_ind + ind_offset

        assigned_gt_inds = temp_assigned_gt_inds[pos_dn_ind]
        assigned_labels = temp_assigned_labels[pos_dn_ind]

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
