# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
import math
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet.models.layers import (MLP, DABDetrTransformerDecoder,
                                 DABDetrTransformerDecoderLayer,
                                 inverse_sigmoid)
from mmengine.model import ModuleList
from torch import Tensor
from typing import List

from .utils import coordinate_to_encoding


class RotatedDABDetrTransformerDecoder(DABDetrTransformerDecoder):
    """Decoder of DAB-DETR.

    Args:
        query_dim (int): The last dimension of query pos,
            5 for rotated anchor format, 4 for anchor format,
            2 for point format. Defaults to 5.
        query_scale_type (str): Type of transformation applied
            to content query. Defaults to `cond_elewise`.
        with_modulated_hw_attn (bool): Whether to inject h&w info
            during cross conditional attention. Defaults to True.
        modulated_with_angle (bool): Whether to reflect angle info
            to h&w in modulated_hw_attn.
    """

    def __init__(self,
                 *args,
                 query_dim: int = 5,
                 modulated_with_angle: bool = False,
                 **kwargs):
        self.modulated_with_angle = modulated_with_angle
        super().__init__(*args, query_dim=query_dim, **kwargs)

    def _init_layers(self):
        """Initialize decoder layers and other layers.

        The code only changed the assert line for angle.
        """
        assert self.query_dim in [2, 4, 5], \
            f'{"dab-detr only supports anchor prior or reference point prior"}'
        assert self.query_scale_type in [
            'cond_elewise', 'cond_scalar', 'fix_elewise'
        ]

        self.layers = ModuleList([
            DABDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims

        self.post_norm = build_norm_layer(self.post_norm_cfg, embed_dims)[1]
        if self.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP(self.query_dim * (embed_dims // 2),
                                  embed_dims, embed_dims, 2)

        if self.with_modulated_hw_attn and self.query_dim >= 4:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 2, 2)

        self.keep_query_pos = self.layers[0].keep_query_pos
        if not self.keep_query_pos:
            for layer_id in range(self.num_layers - 1):
                self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                reg_branches: nn.Module,
                key_padding_mask: Tensor = None,
                **kwargs) -> List[Tensor]:
        """Forward function of decoder. Changes from mmdet include:

            - assert line to angle.
            - coordinate_to_encoding return [...,num_feats * 5] shape.
                for angle encoding
            - with_modulated_hw_attn lines. for reflecting angle info.

        Args:
            query (Tensor): The input query with shape (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim).
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.

        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2/4).
        """
        output = query
        unsigmoid_references = query_pos

        reference_points = unsigmoid_references.sigmoid()
        intermediate_reference_points = [reference_points]

        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            ref_sine_embed = coordinate_to_encoding(
                coord_tensor=obj_center,
                num_feats=self.embed_dims // 2)  # [y,x,w,h,a]
            query_pos = self.ref_point_head(
                ref_sine_embed)  # [bs, nq, 2c] -> [bs, nq, c]
            # For the first decoder layer, do not apply transformation
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            ref_sine_embed = ref_sine_embed[
                ..., :self.embed_dims] * pos_transformation
            # modulated height and weight attention
            if self.with_modulated_hw_attn:
                assert obj_center.size(-1) >= 4
                ref_hw = self.ref_anchor_head(output).sigmoid()
                if self.modulated_with_angle:
                    obj_a = obj_center[..., 4] * math.pi
                    cos = torch.abs(torch.cos(obj_a))
                    sin = torch.sin(obj_a)
                    ref_w = cos * ref_hw[..., 0] + sin * ref_hw[..., 1]
                    ref_h = sin * ref_hw[..., 0] + cos * ref_hw[..., 1]
                    obj_w = cos * obj_center[..., 2] + sin * obj_center[..., 3]
                    obj_h = sin * obj_center[..., 2] + cos * obj_center[..., 3]

                    ref_sine_embed[..., self.embed_dims // 2:] *= \
                        (ref_w / obj_w).unsqueeze(-1)
                    ref_sine_embed[..., :self.embed_dims // 2] *= \
                        (ref_h / obj_h).unsqueeze(-1)
                else:
                    # x * w
                    ref_sine_embed[..., self.embed_dims // 2:] *= \
                        (ref_hw[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                    # y * h
                    ref_sine_embed[..., :self.embed_dims // 2] *= \
                        (ref_hw[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            output = layer(
                output,
                key,
                query_pos=query_pos,
                ref_sine_embed=ref_sine_embed,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                is_first=(layer_id == 0),
                **kwargs)
            # iter update
            tmp_reg_preds = reg_branches(output)
            tmp_reg_preds[..., :self.query_dim] += inverse_sigmoid(
                reference_points)
            new_reference_points = tmp_reg_preds[
                ..., :self.query_dim].sigmoid()
            if layer_id != self.num_layers - 1:
                intermediate_reference_points.append(new_reference_points)
            reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.post_norm(output))

        output = self.post_norm(output)

        if self.return_intermediate:
            return [
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
            ]
        else:
            return [
                output.unsqueeze(0),
                torch.stack(intermediate_reference_points)
            ]
