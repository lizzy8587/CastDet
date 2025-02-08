# Ref: https://github.com/SIAnalytics/RHINO

# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
from mmdet.models.detectors.dab_detr import DABDETR
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer import DABDetrTransformerEncoder
from torch import nn

from mmrotate.registry import MODELS
from ..layers import RotatedDABDetrTransformerDecoder


@MODELS.register_module()
class RotatedDABDETR(DABDETR):
    r"""Angle refine version of DAB-DETR:

    Code is modified from the mmdet.
    """

    def _init_layers(self) -> None:
        """Initialize decoder as RotatedDABDetrTransformerDecoder for angle
        refine."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DABDetrTransformerEncoder(**self.encoder)
        self.decoder = RotatedDABDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_dim = self.decoder.query_dim
        self.query_embedding = nn.Embedding(self.num_queries, self.query_dim)
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
