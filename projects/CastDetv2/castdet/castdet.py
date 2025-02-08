#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   castdet.py
@Version      :   1.0
@Time         :   2024/09/07 16:09:07
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   Implementation of Rotated CastDet.
'''

import copy
import cv2
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, Resize, Compose, CenterCrop

from mmengine.structures import InstanceData
from mmcv.ops import nms_rotated

from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_project
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models.utils.misc import unpack_gt_instances

from mmrotate.registry import MODELS
from mmrotate.models import RotatedSoftTeacher, RotatedSemiBaseDetector
from mmrotate.structures.bbox import rbox_project, RotatedBoxes
from mmrotate.models.utils import filter_gt_instances, _filter_rpn_results_by_score

@MODELS.register_module()
class RotatedCastDet(RotatedSoftTeacher):
    def __init__(self,
                 detector: ConfigType,
                 visual: ConfigType,
                 pseudo_queue_cfg: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(detector, semi_train_cfg, semi_test_cfg, data_preprocessor, init_cfg, **kwargs)

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
        self.clip_logit_scale = semi_train_cfg.get('clip_logit_scale', 100.0)
        self.semi_reg_iter = semi_train_cfg.get('semi_reg_iter', 1)
        self.start_semi_iter = semi_train_cfg.get('start_semi_iter', 0)
        self.start_unsup_iter = semi_train_cfg.get('start_unsup_iter', 0)
        self.initial_rpn_score_thr = semi_train_cfg.get('initial_rpn_score_thr', 0.3)
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

    
    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        
        assert len(multi_batch_data_samples['unsup_teacher'][0].gt_instances) == 0
        assert len(multi_batch_data_samples['unsup_student'][0].gt_instances) == 0

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
        if self.pseudo_queue.cur_iter >= self.start_unsup_iter:
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['unsup_student'],
                multi_batch_data_samples['unsup_student'], batch_info))
        if self.pseudo_queue.cur_iter >= self.start_semi_iter:
            losses.update(**self.loss_by_semi_instances())

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
                    if self.rpn_bbox_type == 'xywh':
                        rois_ = bbox2roi([res.bboxes.convert_to('hbox') for res in rpn_results_list])
                    elif self.rpn_bbox_type == 'xywha':
                        rois_ = bbox2roi([res.bboxes.convert_to('rbox') for res in rpn_results_list])
                    else:   # TODO
                        raise NotImplementedError
                    
                    rois = torch.split(rois_, split_per_img)[idx]  
                    bbox_feats = self.teacher.roi_head.bbox_roi_extractor(
                        x[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
                    if self.teacher.roi_head.with_shared_head:
                        bbox_feats = self.teacher.roi_head.shared_head(bbox_feats)

                    reg_bboxes = self.teacher.roi_head.bbox_head.forward_reg(bbox_feats)
                    img_shape = batch_data_samples[idx].img_shape
                    rpn_result.bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(rois[:, 1:], reg_bboxes, max_shape=img_shape)

                    b = copy.deepcopy(rpn_result.bboxes)
                    if data_samples.get('homography_matrix', None) is not None:
                        b.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))

                    save_proposals.append(b)

                data_samples_ = copy.deepcopy(data_samples)
                data_samples_.gt_instances = copy.deepcopy(rpn_result)
                data_samples_.gt_instances.bboxes = data_samples_.gt_instances.bboxes.tensor
                reg_uncs_list, angle_uncs_list = self.compute_uncertainty_with_aug(
                x, [data_samples_], with_angle_uncs=True)
            
                rpn_result = self.filter_bboxes(rpn_result, save_proposals, reg_uncs_list[0], angle_uncs_list[0])
                if len(rpn_result) == 0:
                    continue

                crop_batch = self.crop_images(inputs.unsqueeze(0), [rpn_result])
                clip_feature = F.normalize(self.get_clip_features(crop_batch), dim=-1)
                scores, labels = (self.clip_logit_scale * clip_feature @ words.T).softmax(dim=-1).max(dim=-1)
                labels[labels>=num_classes] = num_classes
                ids = scores > self.semi_cls_score
                
                img_path = batch_data_samples[idx].img_path
                bboxes = copy.deepcopy(rpn_result.bboxes)
                bboxes.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))

                self.pseudo_queue.update_pseudo_queue(img_path,
                                                      bboxes[ids].detach().cpu().numpy(),
                                                      labels[ids].detach().cpu().numpy(),
                                                      scores[ids].detach().cpu().numpy())
        
        # convert to rpn bbox type
        if self.rpn_bbox_type == 'xywh':
            for rpn_result in rpn_results_list:
                rpn_result.bboxes = rpn_result.bboxes.convert_to('hbox')
        
        
        # ===================== update the pseudo queue =====================
        rpn_results_list = _filter_rpn_results_by_score(rpn_results_list, self.initial_rpn_score_thr)

        if self.semi_train_cfg.get('semi_mask_loss', False):
            results_list = self.teacher.roi_head.predict(
                x, rpn_results_list, batch_data_samples, rescale=False, with_mask=False)
        else:
            results_list = self.teacher.roi_head.predict(
                x, rpn_results_list, batch_data_samples, rescale=False)

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        reg_uncs_list = self.compute_uncertainty_with_aug(
            x, batch_data_samples)

        for data_samples, reg_uncs in zip(batch_data_samples, reg_uncs_list):
            data_samples.gt_instances['reg_uncs'] = reg_uncs
            data_samples.gt_instances.bboxes = rbox_project(
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
            # prepare data
            results = self.pseudo_queue.transform({'img': cv2.imread(img_path)})
            batch_inputs = results['img'].unsqueeze(0).to(self.device)
            data_samples = self.pseudo_queue.bbox2detSample(img_path=img_path,
                                               **results)
            
            x = self.teacher.extract_feat(batch_inputs)
            rpn_result = self.teacher.rpn_head.predict(x, [data_samples], rescale=False)[0]
            
            save_proposals = []
            for i in range(self.semi_reg_iter):
                if self.rpn_bbox_type == 'xywh':
                    rois = bbox2roi([rpn_result.bboxes.convert_to('hbox')])
                elif self.rpn_bbox_type == 'xywha':
                    rois = bbox2roi([rpn_result.bboxes.convert_to('rbox')])
                else:   # TODO
                    raise NotImplementedError
                
                bbox_feats = self.teacher.roi_head.bbox_roi_extractor(
                    x[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
                if self.teacher.roi_head.with_shared_head:
                    bbox_feats = self.teacher.roi_head.shared_head(bbox_feats)
                bbox_preds = self.teacher.roi_head.bbox_head.forward_reg(bbox_feats)

                img_shape = data_samples.img_shape
                rpn_result.bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(
                    rois[:, 1:], bbox_preds, max_shape=img_shape)
                b = copy.deepcopy(rpn_result.bboxes)
                if data_samples.get('homography_matrix', None) is not None:
                    b.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))
                    
                save_proposals.append(b)

            data_samples.gt_instances = copy.deepcopy(rpn_result)
            data_samples.gt_instances.bboxes = data_samples.gt_instances.bboxes.tensor
            
            reg_uncs_list, angle_uncs_list = self.compute_uncertainty_with_aug(
                x, [data_samples], with_angle_uncs=True)
            
            rpn_result.reg_uncs = reg_uncs_list[0]
            rpn_result.angle_uncs = angle_uncs_list[0]
            rpn_result = self.filter_bboxes(rpn_result, save_proposals, reg_uncs_list[0], angle_uncs_list[0])
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
            bboxes = copy.deepcopy(rpn_result.bboxes)
            if data_samples.get('homography_matrix', None) is not None:
                bboxes.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))
            self.pseudo_queue.update_pseudo_queue(img_path,
                                                  bboxes[ids].detach().cpu().numpy(),
                                                  labels[ids].detach().cpu().numpy(),
                                                  scores[ids].detach().cpu().numpy())
            

    def filter_bboxes(self, rpn_result, save_proposals=None, reg_uncs=None, angle_uncs=None):
        # bbox uncs filter
        cur_filter = None
        if reg_uncs is not None:
            jitter_uncs_filter = reg_uncs < self.semi_train_cfg.get('semi_jitter_uncs_thr', 10000)
            cur_filter = cur_filter & jitter_uncs_filter if cur_filter is not None else jitter_uncs_filter

        if angle_uncs is not None:
            angle_uncs_filter = angle_uncs < self.semi_train_cfg.get('semi_jitter_angle_thr', 10000)
            cur_filter = cur_filter & angle_uncs_filter if cur_filter is not None else angle_uncs_filter

        # reg uncs filter
        if save_proposals is not None:
            last_bboxes = save_proposals[-1]
            # TODO: consider angle jittering?
            bboxes = [proposals.tensor[:, 2:4].unsqueeze(0) for proposals in save_proposals]
            bboxes = torch.cat(bboxes, dim=0)
            w, h = last_bboxes.widths, last_bboxes.heights
            var = torch.var(bboxes, dim=0).sum(dim=-1) / (w**2 + h**2)

            reg_uncs_filter = var < self.semi_train_cfg.get('semi_reg_uncs_thr', 10000)
            cur_filter = cur_filter & reg_uncs_filter if cur_filter is not None else reg_uncs_filter

        # bbox size filter
        size_filter = rpn_result.bboxes.areas > self.semi_min_size
        cur_filter = cur_filter & size_filter if cur_filter is not None else size_filter

        rpn_result = rpn_result[cur_filter]

        # rpn score filter
        score_filter = rpn_result.scores > self.semi_rpn_score
        if score_filter.sum() < self.semi_min_rpn_num:
            rpn_result = rpn_result[:self.semi_min_rpn_num]
        elif score_filter.sum() > self.semi_max_rpn_num:
            rpn_result = rpn_result[:self.semi_max_rpn_num]
        else:
            rpn_result = rpn_result[score_filter]

        # nms post processing
        if self.pseudo_nms:
            nms_dets, ids = nms_rotated(rpn_result.bboxes, rpn_result.scores, self.iou_threshold)
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
        
        if self.semi_train_cfg.get('semi_mask_loss', False):
            for res in sampling_results:
                res.pos_priors = res.pos_priors.convert_to('hbox')
            mask_results = self.student.roi_head.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])
        

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
                coor = self.get_coordinate(bbox, self.semi_crop_ratio, image_size, min_size=self.semi_crop_min_size, square=self.semi_train_cfg.get('semi_crop_square', False))
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
        if isinstance(gt_bbox, RotatedBoxes):
            x0, y0, x1, y1 = gt_bbox.convert_to('hbox').tensor.squeeze().tolist()
        else:
            raise NotImplementedError
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

        return super(RotatedSemiBaseDetector, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )