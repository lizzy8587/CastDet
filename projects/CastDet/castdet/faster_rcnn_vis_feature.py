from mmdet.models import TwoStageDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import SampleList
from torch import Tensor
from mmdet.structures.bbox import bbox2roi
import torch
import os
from datetime import datetime

@MODELS.register_module()
class FasterRCNNvis(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        proposals = [res.gt_instances.bboxes for res in batch_data_samples]
        labels = torch.cat([res.gt_instances.labels for res in batch_data_samples])
        rois = bbox2roi(proposals)
        bbox_feats = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
        if self.roi_head.with_shared_head:
            bbox_feats = self.roi_head.shared_head(bbox_feats)
        cls_features = self.roi_head.bbox_head.forward_cls_feature(bbox_feats)

        save_metadata(labels, cls_features, save_dir='work_dirs/coco_step1_mask-rcnn_full_seen_vis/features')

        return batch_data_samples


def save_metadata(labels, cls_features, save_dir='outputs/features'):
    metadata = {
        'labels': labels.cpu(),
        'features': cls_features.cpu()
    }
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{now}.pth')
    torch.save(metadata, save_path)
