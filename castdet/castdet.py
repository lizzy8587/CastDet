#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   castdet.py
@Version      :   1.0
@Time         :   2024/04/08 21:06:22
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   Implementation of CastDet.
'''

from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_project
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models.utils.misc import unpack_gt_instances
from mmdet.models import SemiBaseDetector
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, Resize, Compose, CenterCrop
from mmcv.ops import nms
import copy
import cv2
from tqdm import tqdm


@MODELS.register_module()
class CastDet(SemiBaseDetector):
    def __init__(self,
                 detector: ConfigType,
                 visual: ConfigType,
                 pseudo_queue_cfg: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(detector, semi_train_cfg, semi_test_cfg, data_preprocessor, init_cfg)

        self.visual = MODELS.build(visual)
        self.pseudo_queue = MODELS.build(pseudo_queue_cfg)
        self.semi_max_rpn_num = semi_train_cfg.get('semi_max_rpn_num', 8)
        self.semi_min_rpn_num = semi_train_cfg.get('semi_min_rpn_num', 0)
        self.semi_rpn_score = semi_train_cfg.get('semi_rpn_score', 0.9)
        self.semi_cls_score = semi_train_cfg.get('semi_cls_score', 0.9)
        self.semi_min_size = semi_train_cfg.get('semi_min_size', 1000)
        self.semi_min_label = semi_train_cfg.get('semi_min_label', 0)
        self.semi_bbox_loss = semi_train_cfg.get('semi_bbox_loss', True)
        self.ignore_bg = semi_train_cfg.get('ignore_bg', False)
        self.unsup_cls_loss = semi_train_cfg.get('unsup_cls_loss', True)
        self.unsup_rpn_loss = semi_train_cfg.get('unsup_rpn_loss', True)
        self.semi_crop_min_size = semi_train_cfg.get('semi_crop_min_size', (200, 200))
        self.semi_crop_ratio = semi_train_cfg.get('semi_crop_ratio', 0.0)
        self.semi_jitter_num = semi_train_cfg.get('semi_jitter_num', -1)
        self.semi_select_jitter_num = semi_train_cfg.get('semi_select_jitter_num', 16)
        self.clip_logit_scale = semi_train_cfg.get('clip_logit_scale', 100.0)
        self.semi_reg_iter = semi_train_cfg.get('semi_reg_iter', 1)
        self.start_semi_iter = semi_train_cfg.get('start_semi_iter', 0)
        vector_path = semi_train_cfg.get('vector_path', None)
        self.words = nn.Parameter(torch.tensor(np.load(vector_path)), requires_grad=False) if vector_path is not None else None

        if semi_train_cfg.get('semi_crop_square', False):
            self.resize = Compose([
                Resize(size=visual.image_size-20, max_size=visual.image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=(visual.image_size, visual.image_size)),
            ])
        else:
            self.resize = Resize(size=(visual.image_size, visual.image_size), interpolation=InterpolationMode.BICUBIC)

        # nms config for pseudo labels
        self.pseudo_nms = semi_train_cfg.get('pseudo_nms', False)
        self.iou_threshold = semi_train_cfg.get('iou_threshold', 0.6)
        self.max_keep = semi_train_cfg.get('max_keep', 100000)

    
    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:

        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

        origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'],
            multi_batch_data_samples['unsup_teacher'])
        multi_batch_data_samples[
            'unsup_student'] = self.project_pseudo_instances(
                origin_pseudo_data_samples,
                multi_batch_data_samples['unsup_student'])
        if self.pseudo_queue.cur_iter >= self.start_semi_iter:
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['unsup_student'],
                multi_batch_data_samples['unsup_student'], batch_info))
            losses.update(**self.loss_by_semi_instances())

        return losses

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        x = self.student.extract_feat(batch_inputs)

        losses = {}
        
        rpn_losses, rpn_results_list = self.rpn_loss_by_pseudo_instances(
            x, batch_data_samples)
        if self.unsup_rpn_loss:
            losses.update(**rpn_losses)
        if self.unsup_cls_loss:
            losses.update(**self.rcnn_cls_loss_by_pseudo_instances(
                x, rpn_results_list, batch_data_samples, batch_info))
        losses.update(**self.rcnn_reg_loss_by_pseudo_instances(
            x, rpn_results_list, batch_data_samples))
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    def rcnn_cls_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                          unsup_rpn_results_list: InstanceList,
                                          batch_data_samples: SampleList,
                                          batch_info: dict) -> dict:
        rpn_results_list = copy.deepcopy(unsup_rpn_results_list)
        cls_data_samples = copy.deepcopy(batch_data_samples)
        cls_data_samples = filter_gt_instances(
            cls_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)

        outputs = unpack_gt_instances(cls_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(cls_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')
            assign_result = self.student.roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        selected_bboxes = [res.priors for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(x, rois)
        # cls_reg_targets is a tuple of labels, label_weights,
        # and bbox_targets, bbox_weights
        cls_reg_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, self.student.train_cfg.rcnn)

        selected_results_list = []
        for bboxes, data_samples, teacher_matrix, teacher_img_shape in zip(
                selected_bboxes, batch_data_samples,
                batch_info['homography_matrix'], batch_info['img_shape']):
            student_matrix = torch.tensor(
                data_samples.homography_matrix, device=teacher_matrix.device)
            homography_matrix = teacher_matrix @ student_matrix.inverse()
            projected_bboxes = bbox_project(bboxes, homography_matrix,
                                            teacher_img_shape)
            selected_results_list.append(InstanceData(bboxes=projected_bboxes))

        with torch.no_grad():
            if self.semi_train_cfg.get('bg_soft_score', 'student'):
                results_list = self.student.roi_head.predict_bbox(
                    batch_info['feat'],
                    batch_info['metainfo'],
                    selected_results_list,
                    rcnn_test_cfg=None,
                    rescale=False)
            else:
                results_list = self.teacher.roi_head.predict_bbox(
                    batch_info['feat'],
                    batch_info['metainfo'],
                    selected_results_list,
                    rcnn_test_cfg=None,
                    rescale=False)
            bg_score = torch.cat(
                [results.scores[:, -1] for results in results_list])
            # cls_reg_targets[0] is labels
            neg_inds = cls_reg_targets[
                0] == self.student.roi_head.bbox_head.num_classes
            # cls_reg_targets[1] is label_weights
            cls_reg_targets[1][neg_inds] = bg_score[neg_inds].detach()

        losses = self.student.roi_head.bbox_head.loss(
            bbox_results['cls_score'], bbox_results['bbox_pred'], rois,
            *cls_reg_targets)
        # cls_reg_targets[1] is label_weights
        if 'loss_cls' in losses.keys():
            losses['loss_cls'] = losses['loss_cls'] * len(
                cls_reg_targets[1]) / max(sum(cls_reg_targets[1]), 1.0)
        return losses

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'
        x = self.teacher.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # ===================== update the pseudo queue =====================
        if self.pseudo_queue.cur_iter==0:
            self.initialize_pseudo_queue()
            self.pseudo_queue.save_queue()
            
        if self.pseudo_queue.start_update():
            num_classes = self.teacher.roi_head.bbox_head.num_classes
            if self.words is None:
                words = F.normalize(self.teacher.roi_head.bbox_head.fc_cls.get_words(with_bg=True), dim=-1)
            else:
                words = F.normalize(self.words, dim=-1)

            split_per_img = tuple([len(res.labels) for res in rpn_results_list])

            for idx, (inputs, rpn_result, data_samples) in enumerate(zip(batch_inputs, rpn_results_list, batch_data_samples)):
                save_proposals = []
                for i in range(self.semi_reg_iter):
                    rois_ = bbox2roi([res.bboxes for res in rpn_results_list])
                    rois = torch.split(rois_, split_per_img)[idx]  
                    bbox_feats = self.teacher.roi_head.bbox_roi_extractor(
                        x[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
                    if self.teacher.roi_head.with_shared_head:
                        bbox_feats = self.teacher.roi_head.shared_head(bbox_feats)

                    reg_bboxes = self.teacher.roi_head.bbox_head.forward_reg(bbox_feats)
                    img_shape = batch_data_samples[idx].img_shape
                    rpn_result.bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(rois[:, 1:], reg_bboxes, max_shape=img_shape)

                    b = rpn_result.bboxes
                    if data_samples.get('homography_matrix', None) is not None:
                        b = bbox_project(
                        rpn_result.bboxes, torch.from_numpy(data_samples.homography_matrix).inverse().to(
                            self.data_preprocessor.device), data_samples.ori_shape)
                    save_proposals.append(b)


                rpn_result = self.filter_bboxes(rpn_result, save_proposals)
                if len(rpn_result) == 0:
                    continue

                crop_batch = self.crop_images(inputs.unsqueeze(0), [rpn_result])
                clip_feature = F.normalize(self.get_clip_features(crop_batch), dim=-1)
                scores, labels = (self.clip_logit_scale * clip_feature @ words.T).softmax(dim=-1).max(dim=-1)
                labels[labels>=num_classes] = num_classes
                ids = scores > self.semi_cls_score
                
                img_path = batch_data_samples[idx].img_path
                bboxes = bbox_project(
                    rpn_result.bboxes, torch.from_numpy(data_samples.homography_matrix).inverse().to(
                        self.data_preprocessor.device), data_samples.ori_shape)
                self.pseudo_queue.update_pseudo_queue(img_path,
                                                      bboxes[ids].detach().cpu().numpy(),
                                                      labels[ids].detach().cpu().numpy(),
                                                      scores[ids].detach().cpu().numpy())
        # ===================== update the pseudo queue =====================

        results_list = self.teacher.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=False)

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        batch_data_samples = filter_gt_instances(
            batch_data_samples,
            score_thr=self.semi_train_cfg.pseudo_label_initial_score_thr)

        reg_uncs_list = self.compute_uncertainty_with_aug(
            x, batch_data_samples)

        for data_samples, reg_uncs in zip(batch_data_samples, reg_uncs_list):
            data_samples.gt_instances['reg_uncs'] = reg_uncs
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)

        batch_info = {
            'feat': x,
            'img_shape': [],
            'homography_matrix': [],
            'metainfo': []
        }
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.from_numpy(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)
        return batch_data_samples, batch_info


    def initialize_pseudo_queue(self):
        for img_path in tqdm(self.pseudo_queue.init_imgs):
            results = self.pseudo_queue.transform({'img': cv2.imread(img_path)})
            batch_inputs = results['img'].unsqueeze(0).to(self.device)
            data_samples = self.pseudo_queue.bbox2detSample(img_path=img_path,
                                               **results)
            
            x = self.teacher.extract_feat(batch_inputs)
            rpn_result = self.teacher.rpn_head.predict(x, [data_samples], rescale=False)[0]
            save_proposals = []
            for i in range(self.semi_reg_iter):
                rois = bbox2roi([rpn_result.bboxes])
                bbox_feats = self.teacher.roi_head.bbox_roi_extractor(
                    x[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
                if self.teacher.roi_head.with_shared_head:
                    bbox_feats = self.teacher.roi_head.shared_head(bbox_feats)
                bbox_preds = self.teacher.roi_head.bbox_head.forward_reg(bbox_feats)

                img_shape = data_samples.img_shape
                rpn_result.bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(
                    rois[:, 1:], bbox_preds, max_shape=img_shape)
                b = rpn_result.bboxes
                if data_samples.get('homography_matrix', None) is not None:
                    b = bbox_project(
                    rpn_result.bboxes, torch.from_numpy(data_samples.homography_matrix).inverse().to(
                        self.data_preprocessor.device), data_samples.ori_shape)
                save_proposals.append(b)

            rpn_result = self.filter_bboxes(rpn_result, save_proposals)
            if len(rpn_result) == 0:
                continue

            num_classes = self.teacher.roi_head.bbox_head.num_classes
            if self.words is None:
                words = F.normalize(self.teacher.roi_head.bbox_head.fc_cls.get_words(with_bg=True), dim=-1)
            else:
                words = F.normalize(self.words, dim=-1)

            crop_batch = self.crop_images(batch_inputs, [rpn_result])
            clip_feature = F.normalize(self.get_clip_features(crop_batch), dim=-1)
            scores, labels = (self.clip_logit_scale * clip_feature @ words.T).softmax(dim=-1).max(dim=-1)
            labels[labels>=num_classes] = num_classes
            ids = scores > self.semi_cls_score
            
            img_path = data_samples.img_path
            if data_samples.get('homography_matrix', None) is not None:
                rpn_result.bboxes = bbox_project(
                    rpn_result.bboxes, torch.from_numpy(data_samples.homography_matrix).inverse().to(
                        self.data_preprocessor.device), data_samples.ori_shape)
            self.pseudo_queue.update_pseudo_queue(img_path,
                                                  rpn_result.bboxes[ids].detach().cpu().numpy(),
                                                  labels[ids].detach().cpu().numpy(),
                                                  scores[ids].detach().cpu().numpy())


    def filter_bboxes(self, rpn_result, save_proposals=None):
        if len(rpn_result)==0:
            return rpn_result
        if save_proposals is not None and self.semi_jitter_num > 0:
            last_bboxes = save_proposals[-1][:self.semi_jitter_num]
            bboxes = [proposals[:self.semi_jitter_num].unsqueeze(0) for proposals in save_proposals]
            bboxes = torch.cat(bboxes, dim=0)
            w, h = last_bboxes[:,2] - last_bboxes[:,0], last_bboxes[:,3] - last_bboxes[:,1]
            var = torch.var(bboxes, dim=0).sum(dim=-1) / (w**2 + h**2)
            
            v, ids = torch.topk(var, self.semi_select_jitter_num, largest=False)
            ids, _ = ids.sort()
            rpn_result = rpn_result[ids]
        semi_max_rpn_num = min(len(rpn_result.scores), self.semi_max_rpn_num)
        semi_min_rpn_num = min(len(rpn_result.scores), self.semi_min_rpn_num)
        r, l = rpn_result.scores[semi_min_rpn_num - 1], rpn_result.scores[semi_max_rpn_num - 1]
        score_thr = min(r, max(l, self.semi_rpn_score))
        score_filter = rpn_result.scores > score_thr
        
        w = rpn_result.bboxes[:,2]-rpn_result.bboxes[:,0]
        h = rpn_result.bboxes[:,3]-rpn_result.bboxes[:,1]
        size_filter = (w * h) > self.semi_min_size
            
        rpn_result = rpn_result[score_filter & size_filter]

        if self.pseudo_nms and len(rpn_result) > self.max_keep:
            nms_dets, ids = nms(rpn_result.bboxes, rpn_result.scores, self.iou_threshold, max_num=self.max_keep)
            rpn_result = rpn_result[ids]

        return rpn_result
    

    def loss_by_semi_instances(self) -> dict:
        
        """Get pseudo instances from pseudo queue."""
        batch_inputs, batch_data_samples = self.pseudo_queue.get_image_detsample()
        num_imgs = len(batch_data_samples)
        batch_gt_instances = [det_sample.gt_instances for det_sample in batch_data_samples]

        if num_imgs == 0:
            return dict()
        assert self.student.with_bbox, 'Bbox head must be implemented.'
        x = self.student.extract_feat(batch_inputs)

        # rpn_results_list = self.student.rpn_head.predict(x, batch_data_samples, rescale=False)
        rpn_losses, rpn_results_list = self.student.rpn_head.loss_and_predict(x, batch_data_samples)

        # assign gts and sample proposals
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.student.roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        
        # ===================== into >> `bbox_loss` =====================
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self.student.roi_head._bbox_forward(x, rois)

        labels, label_weights, bbox_targets, bbox_weights = self.student.roi_head.bbox_head.get_targets(
            sampling_results, self.student.roi_head.train_cfg, concat=True)

        if isinstance(self.semi_min_label, int):
            ids = labels >= self.semi_min_label
        else:
            ids = torch.isin(labels, torch.tensor(self.semi_min_label).cuda())

        if self.ignore_bg:
            # label_weights[labels==self.student.roi_head.bbox_head.num_classes] = 0
            ids = ids & (labels!=self.student.roi_head.bbox_head.num_classes)

        bbox_results['cls_score'] = bbox_results['cls_score'][ids]
        bbox_results['bbox_pred'] = bbox_results['bbox_pred'][ids]
        rois = rois[ids]
        labels = labels[ids]
        label_weights = label_weights[ids]
        bbox_targets = bbox_targets[ids]
        bbox_weights = bbox_weights[ids]

        semi_bg_weight = self.semi_train_cfg.get('semi_bg_weight', 1.0)
        label_weights[labels==self.student.roi_head.bbox_head.num_classes] = semi_bg_weight

        losses = self.student.roi_head.bbox_head.loss(
                            bbox_results['cls_score'],
                            bbox_results['bbox_pred'] if self.semi_bbox_loss else None,
                            rois,
                            labels,
                            label_weights,
                            bbox_targets,
                            bbox_weights,
                            reduction_override=None)
        semi_weight = self.semi_train_cfg.get('semi_weight', 1.)
        if self.semi_train_cfg.get('semi_rpn_loss', False):
            losses.update(rename_loss_dict('semi_', rpn_losses))
        losses = rename_loss_dict('semi_', reweight_loss_dict(losses, semi_weight))

        return losses


    def crop_images(self, batch_inputs: Tensor,
                    rpn_results_list: InstanceList,) -> Tensor:
        batch_crops = []
        for batch, results in zip(batch_inputs, rpn_results_list):
            image_size = (batch.shape[-1], batch.shape[-2])
            for bbox in results.bboxes:
                coor = self.get_coordinate(bbox, self.semi_crop_ratio, image_size, min_size=self.semi_crop_min_size)
                # coor = self.get_coordinate(bbox, 0.0, image_size, min_size=(200,200))
                batch_crops.append(self.resize(batch[:,coor[1]:coor[3],coor[0]:coor[2]])[None,:])
        batch_crops = torch.cat(batch_crops, dim=0)
        return batch_crops

    @torch.no_grad()
    def get_clip_features(self, batch_inputs: Tensor) -> Tensor:
        self.visual.eval()
        return self.visual(batch_inputs)
    
    def get_coordinate(self, gt_bbox, ratio=0, image_size=(800, 800), min_size=(-1, -1), square=False):
        # image_size(w, h)
        x0, y0, x1, y1 = gt_bbox
        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + w//2, y0 + h//2
        
        # w = h = max(w*(1+ratio), h*(1+ratio), min_size[0], min_size[1])
        w = min(image_size[0], max(w*(1+ratio), min_size[0]))
        h = min(image_size[1], max(h*(1+ratio), min_size[1]))
        if square:
            w = h = min(image_size[0], image_size[1], max(w, h))
        
        cx = w//2 if cx < w//2 else cx
        cx = image_size[0]-w//2 if cx > image_size[0]-w//2 else cx
        cy = h//2 if cy < h//2 else cy
        cy = image_size[1]-h//2 if cy > image_size[1]-h//2 else cy
        x0, x1, y0, y1 = int(cx-w//2), int(cx+w//2), int(cy-h//2), int(cy+h//2)

        return (x0, y0, x1, y1)

    @property
    def device(self) -> torch.device:
        return self.words.device

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:

        return super(SemiBaseDetector, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    
    def rcnn_reg_loss_by_pseudo_instances(
            self, x: Tuple[Tensor], unsup_rpn_results_list: InstanceList,
            batch_data_samples: SampleList) -> dict:
        """Calculate rcnn regression loss from a batch of inputs and pseudo
        data samples.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            unsup_rpn_results_list (list[:obj:`InstanceData`]):
                List of region proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.

        Returns:
            dict[str, Tensor]: A dictionary of rcnn
                regression loss components
        """
        rpn_results_list = copy.deepcopy(unsup_rpn_results_list)
        reg_data_samples = copy.deepcopy(batch_data_samples)
        for data_samples in reg_data_samples:
            if data_samples.gt_instances.bboxes.shape[0] > 0:
                data_samples.gt_instances = data_samples.gt_instances[
                    data_samples.gt_instances.reg_uncs <
                    self.semi_train_cfg.reg_pseudo_thr]
        roi_losses = self.student.roi_head.loss(x, rpn_results_list,
                                                reg_data_samples)
        return {'loss_bbox': roi_losses['loss_bbox']}

    def rpn_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                     batch_data_samples: SampleList) -> dict:
        """Calculate rpn loss from a batch of inputs and pseudo data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
        Returns:
            dict: A dictionary of rpn loss components
        """

        rpn_data_samples = copy.deepcopy(batch_data_samples)
        rpn_data_samples = filter_gt_instances(
            rpn_data_samples, score_thr=self.semi_train_cfg.rpn_pseudo_thr)
        proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                  self.student.test_cfg.rpn)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.student.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        for key in rpn_losses.keys():
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        return rpn_losses, rpn_results_list


    def compute_uncertainty_with_aug(
            self, x: Tuple[Tensor],
            batch_data_samples: SampleList) -> List[Tensor]:
        """Compute uncertainty with augmented bboxes.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.

        Returns:
            list[Tensor]: A list of uncertainty for pseudo bboxes.
        """
        auged_results_list = self.aug_box(batch_data_samples,
                                          self.semi_train_cfg.jitter_times,
                                          self.semi_train_cfg.jitter_scale)
        # flatten
        auged_results_list = [
            InstanceData(bboxes=auged.reshape(-1, auged.shape[-1]))
            for auged in auged_results_list
        ]

        self.teacher.roi_head.test_cfg = None
        results_list = self.teacher.roi_head.predict(
            x, auged_results_list, batch_data_samples, rescale=False)
        self.teacher.roi_head.test_cfg = self.teacher.test_cfg.rcnn

        reg_channel = max(
            [results.bboxes.shape[-1] for results in results_list]) // 4
        bboxes = [
            results.bboxes.reshape(self.semi_train_cfg.jitter_times, -1,
                                   results.bboxes.shape[-1])
            if results.bboxes.numel() > 0 else results.bboxes.new_zeros(
                self.semi_train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for results in results_list
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel,
                             4)[torch.arange(bbox.shape[0]), label]
                for bbox, label in zip(bboxes, labels)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel,
                            4)[torch.arange(unc.shape[0]), label]
                for unc, label in zip(box_unc, labels)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0)
                     for bbox in bboxes]
        box_unc = [
            torch.mean(
                unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4), dim=-1)
            if wh.numel() > 0 else unc for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(batch_data_samples, times, frac):
        """Augment bboxes with jitter."""

        def _aug_single(box):
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2,
                                                          2).reshape(-1, 4))
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device) *
                aug_scale[None, ...])
            new_box = box.clone()[None, ...].expand(times, box.shape[0],
                                                    -1) + offset
            return new_box

        return [
            _aug_single(data_samples.gt_instances.bboxes)
            for data_samples in batch_data_samples
        ]