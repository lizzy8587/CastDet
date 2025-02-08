# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
from mmcv.cnn import Linear
from mmdet.models.dense_heads import DABDETRHead
from mmdet.models.layers import MLP

from mmrotate.registry import MODELS
from .rotated_conditional_detr_head import RotatedConditionalDETRHead


@MODELS.register_module()
class RotatedDABDETRHead(DABDETRHead, RotatedConditionalDETRHead):
    """Rotated version of Head of DAB-DETR.

    Methods are inherited as follows.
        DABDETRHead
            - init_weights
            - forward
            - predict
        RotatedConditionalDETRHead
            - other methods
    """

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.fc_reg = MLP(self.embed_dims, self.embed_dims, self.reg_dim, 3)
