# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
from mmdet.models.dense_heads import ConditionalDETRHead

from mmrotate.registry import MODELS
from .rotated_detr_head import RotatedDETRHead


@MODELS.register_module()
class RotatedConditionalDETRHead(ConditionalDETRHead, RotatedDETRHead):
    """Rotated version of Head of Conditional DETR.

    Methods are inherited as follows.
        ConditionalDETRHead
            - init_weights
            - forward
            - loss
            - loss_and_predict
            - predict
        RotatedConditionalDETRHead
            - other methods
    """
    pass
