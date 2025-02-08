# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
import math
import torch
from torch import Tensor


def rotated_coordinate_to_encoding(coord_tensor: Tensor,
                                   num_feats: int = 128,
                                   temperature: int = 10000,
                                   scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Added encoding for coord_tensor.size(-1) == 5 from mmdet.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. The last dimension should be 5.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    assert coord_tensor.size(-1) == 5, \
        'coord_tensor.size(-1) should be 5, but got {}'.format(
            coord_tensor.size(-1))
    # project angle to the point on the unit circle
    theta = coord_tensor[..., 4]
    angle_cx, angle_cy = torch.cos(2 * theta), torch.sin(2 * theta)

    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)

    w_embed = coord_tensor[..., 2] * scale
    pos_w = w_embed[..., None] / dim_t
    pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                        dim=-1).flatten(2)

    h_embed = coord_tensor[..., 3] * scale
    pos_h = h_embed[..., None] / dim_t
    pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                        dim=-1).flatten(2)

    angle_cx = angle_cx[..., None] / dim_t
    angle_cy = angle_cy[..., None] / dim_t
    pos_a = torch.stack((angle_cx[..., 0::2].sin(), angle_cx[..., 1::2].cos(),
                         angle_cy[..., 0::2].sin(), angle_cy[..., 1::2].cos()),
                        dim=-1).flatten(2)

    pos = torch.cat((pos_y, pos_x, pos_w, pos_h, pos_a), dim=-1)

    return pos


def coordinate_to_encoding(coord_tensor: Tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Added encoding for coord_tensor.size(-1) == 5 from mmdet.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4 or 5.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    elif coord_tensor.size(-1) == 5:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        a_embed = coord_tensor[..., 4] * scale
        pos_a = a_embed[..., None] / dim_t
        pos_a = torch.stack((pos_a[..., 0::2].sin(), pos_a[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h, pos_a), dim=-1)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            coord_tensor.size(-1)))
    return pos
