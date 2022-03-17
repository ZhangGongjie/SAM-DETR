# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.misc import _get_clones, _get_activation_fn


class TransformerEncoder(nn.Module):
    def __init__(self, args, encoder_layer, num_layers):
        super().__init__()
        self.args = args
        self.num_layers = num_layers
        self.layers = _get_clones(encoder_layer, num_layers)
        assert num_layers == self.args.enc_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args, activation="relu"):
        super().__init__()
        self.args = args
        self.d_model = args.hidden_dim
        self.nheads = args.nheads
        self.num_queries = args.num_queries
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout
        self.activation = _get_activation_fn(activation)

        # Encoder Self-Attention
        self.self_attn = nn.MultiheadAttention(self.d_model, self.nheads, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # FFN
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout2 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.dropout3 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # Self-Attention
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, args, encoder_layer, num_layers):
        super().__init__()
        self.args = args
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        assert num_layers == self.args.enc_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, args, activation='relu'):
        super().__init__()

        self.args = args
        self.d_model = args.hidden_dim
        self.nheads = args.nheads
        self.num_queries = args.num_queries
        # Note: Multiscale encoder's dim_feedforward halved for memory efficiency
        self.dim_feedforward = args.dim_feedforward // 2
        self.dropout = args.dropout

        # Hard-coded Hyper-parameters
        self.n_feature_levels = 3
        self.n_points = 4

        # self attention
        from models.ops.modules import MSDeformAttn
        self.self_attn = MSDeformAttn(self.d_model, self.n_feature_levels, self.nheads, self.n_points)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # ffn
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.dropout3 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos),
                              reference_points,
                              src,
                              spatial_shapes,
                              level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src
