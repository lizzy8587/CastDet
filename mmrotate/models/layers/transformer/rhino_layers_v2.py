# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
import math
import torch
from mmdet.models.layers.transformer import (MLP, DinoTransformerDecoder,
                                             inverse_sigmoid)
from mmengine.model import ModuleList
from torch import Tensor, nn

from .rotated_attention import RotatedDeformableDetrTransformerDecoderLayer
from .utils import coordinate_to_encoding, rotated_coordinate_to_encoding


class RhinoTransformerDecoderV2(DinoTransformerDecoder):
    """Transformer encoder of RHINO.

    Unlike the original one, it uses 5d references without dropping angle.
    """

    def __init__(self, *args, angle_factor=math.pi, **kwargs):
        self.angle_factor = angle_factor
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            RotatedDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self,
                query: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                self_attn_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: nn.ModuleList,
                return_sampling_results: bool = False,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (num_queries, bs, dim)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]

        if return_sampling_results:
            sampling_locations = []
            sampling_offsets = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 5:
                dummy_angle = torch.ones_like(
                    valid_ratios[..., :1]) * self.angle_factor
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios, dummy_angle], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            # skip angle dim
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :4])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                return_sampling_results=return_sampling_results,
                **kwargs)

            if return_sampling_results:
                sampling_locations.append(layer.cross_attn.sampling_locs)
                sampling_offsets.append(layer.cross_attn.sampling_offs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)

                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            intermediate = torch.stack(intermediate)
            intermediate_reference_points = torch.stack(
                intermediate_reference_points)

            if return_sampling_results:
                return (intermediate, intermediate_reference_points,
                        sampling_locations, sampling_offsets)
            return intermediate, intermediate_reference_points

        return query, reference_points


class RhinoTransformerDecoderV4(DinoTransformerDecoder):
    """Transformer encoder of RHINO.

    This decoder encodes 5d reference points to embedding without dropping
    angle. It encodes angle as sin(x), cos(y).
    """

    def __init__(self, *args, angle_factor=math.pi, **kwargs):
        self.angle_factor = angle_factor
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            RotatedDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 3, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self,
                query: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                self_attn_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: nn.ModuleList,
                return_sampling_results: bool = False,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (num_queries, bs, dim)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]

        if return_sampling_results:
            sampling_locations = []
            sampling_offsets = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 5:
                dummy_angle = torch.ones_like(
                    valid_ratios[..., :1]) * self.angle_factor
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios, dummy_angle], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            # skip angle dim
            query_sine_embed = rotated_coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                return_sampling_results=return_sampling_results,
                **kwargs)

            if return_sampling_results:
                sampling_locations.append(layer.cross_attn.sampling_locs)
                sampling_offsets.append(layer.cross_attn.sampling_offs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)

                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            intermediate = torch.stack(intermediate)
            intermediate_reference_points = torch.stack(
                intermediate_reference_points)

            if return_sampling_results:
                return (intermediate, intermediate_reference_points,
                        sampling_locations, sampling_offsets)
            return intermediate, intermediate_reference_points

        return query, reference_points
