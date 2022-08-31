# # Copyright (c) OpenMMLab. All rights reserved.
# import math
# import warnings

# ############
# #from torch.utils.checkpoint import checkpoint
# #############

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
#                       constant_init, normal_init, trunc_normal_init)
# from mmcv.cnn.bricks.drop import build_dropout
# from mmcv.cnn.bricks.transformer import MultiheadAttention
# from mmcv.cnn.utils.weight_init import trunc_normal_
# from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
#                          load_state_dict)
# from torch.nn.modules.utils import _pair as to_2tuple

# from ...utils import get_root_logger
# from ..builder import BACKBONES
# from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, pvt_convert


# class MixFFN(BaseModule):
#     """An implementation of MixFFN of PVT.

#      The differences between MixFFN & FFN:
#         1. Use 1X1 Conv to replace Linear layer.
#         2. Introduce 3X3 Depth-wise Conv to encode positional information.

#      Args:
#         embed_dims (int): The feature dimension. Same as
#             `MultiheadAttention`.
#         feedforward_channels (int): The hidden dimension of FFNs.
#         act_cfg (dict, optional): The activation config for FFNs.
#             Default: dict(type='GELU').
#         ffn_drop (float, optional): Probability of an element to be
#             zeroed in FFN. Default 0.0.
#         dropout_layer (obj:`ConfigDict`): The dropout_layer used
#             when adding the shortcut.
#             Default: None.
#         use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
#             Defaults: False.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#     """

#     def __init__(self,
#                  embed_dims,
#                  feedforward_channels,
#                  act_cfg=dict(type='GELU'),
#                  ffn_drop=0.,
#                  dropout_layer=None,
#                  use_conv=False,
#                  init_cfg=None):
#         super(MixFFN, self).__init__(init_cfg=init_cfg)

#         self.embed_dims = embed_dims
#         self.feedforward_channels = feedforward_channels
#         self.act_cfg = act_cfg
#         activate = build_activation_layer(act_cfg)

#         in_channels = embed_dims
#         fc1 = Conv2d(
#             in_channels=in_channels,
#             out_channels=feedforward_channels,
#             kernel_size=1,
#             stride=1,
#             bias=True)
#         if use_conv:
#             # 3x3 depth wise conv to provide positional encode information
#             dw_conv = Conv2d(
#                 in_channels=feedforward_channels,
#                 out_channels=feedforward_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=(3 - 1) // 2,
#                 bias=True,
#                 groups=feedforward_channels)
#         fc2 = Conv2d(
#             in_channels=feedforward_channels,
#             out_channels=in_channels,
#             kernel_size=1,
#             stride=1,
#             bias=True)
#         drop = nn.Dropout(ffn_drop)
#         layers = [fc1, activate, drop, fc2, drop]
#         if use_conv:
#             layers.insert(1, dw_conv)
#         self.layers = Sequential(*layers)
#         self.dropout_layer = build_dropout(
#             dropout_layer) if dropout_layer else torch.nn.Identity()

#     def forward(self, x, hw_shape, identity=None):
#         out = nlc_to_nchw(x, hw_shape)
#         out = self.layers(out)
#         out = nchw_to_nlc(out)
#         if identity is None:
#             identity = x
#         return identity + self.dropout_layer(out)


# class SpatialReductionAttention(MultiheadAttention):
#     """An implementation of Spatial Reduction Attention of PVT.

#       This module is modified from MultiheadAttention which is a module from
#       mmcv.cnn.bricks.transformer.

#      Args:
#         embed_dims (int): The embedding dimension.
#         num_heads (int): Parallel attention heads.
#         attn_drop (float): A Dropout layer on attn_output_weights.
#             Default: 0.0.
#         proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
#             Default: 0.0.
#         dropout_layer (obj:`ConfigDict`): The dropout_layer used
#             when adding the shortcut. Default: None.
#         batch_first (bool): Key, Query and Value are shape of
#             (batch, n, embed_dim)
#             or (n, batch, embed_dim). Default: False.
#         qkv_bias (bool): enable bias for qkv if True. Default: True.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN').
#         sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
#             Attention of PVT. Default: 1.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#     """

#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  attn_drop=0.,
#                  proj_drop=0.,
#                  dropout_layer=None,
#                  batch_first=True,
#                  qkv_bias=True,
#                  norm_cfg=dict(type='LN'), #
#                  sr_ratio=1, #
#                  init_cfg=None):
#         super().__init__(
#             embed_dims,
#             num_heads,
#             attn_drop,
#             proj_drop,
#             batch_first=batch_first,
#             dropout_layer=dropout_layer,
#             bias=qkv_bias,
#             init_cfg=init_cfg)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = Conv2d(
#                 in_channels=embed_dims,
#                 out_channels=embed_dims,
#                 kernel_size=sr_ratio,
#                 stride=sr_ratio)
#             # The ret[0] of build_norm_layer is norm name.
#             self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

#         # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
#         from mmdet import digit_version, mmcv_version
#         if mmcv_version < digit_version('1.3.17'):
#             warnings.warn('The legacy version of forward function in'
#                           'SpatialReductionAttention is deprecated in'
#                           'mmcv>=1.3.17 and will no longer support in the'
#                           'future. Please upgrade your mmcv.')
#             self.forward = self.legacy_forward

#     def forward(self, x, hw_shape, identity=None):

#         x_q = x
#         if self.sr_ratio > 1:
#             x_kv = nlc_to_nchw(x, hw_shape)
#             x_kv = self.sr(x_kv)
#             x_kv = nchw_to_nlc(x_kv)
#             x_kv = self.norm(x_kv)
#         else:
#             x_kv = x

#         if identity is None:
#             identity = x_q

#         """ say:
#             # Because the dataflow('key', 'query', 'value') of
#             # ``torch.nn.MultiheadAttention`` is (num_query, batch,
#             # embed_dims), We should adjust the shape of dataflow from
#             # batch_first (batch, num_query, embed_dims) to num_query_first
#             # (num_query ,batch, embed_dims), and recover ``attn_output``
#             # from num_query_first to batch_first.
#         """

#         if self.batch_first:
#             x_q = x_q.transpose(0, 1)   # Batch query dim ->query Batch dim
#             x_kv = x_kv.transpose(0, 1)

#         out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]  #继承自MultiheadAttention

#         if self.batch_first:
#             out = out.transpose(0, 1)

#         return identity + self.dropout_layer(self.proj_drop(out))

#     def legacy_forward(self, x, hw_shape, identity=None):
#         """multi head attention forward in mmcv version < 1.3.17."""
#         x_q = x
#         if self.sr_ratio > 1:
#             x_kv = nlc_to_nchw(x, hw_shape)
#             x_kv = self.sr(x_kv)
#             x_kv = nchw_to_nlc(x_kv)
#             x_kv = self.norm(x_kv)
#         else:
#             x_kv = x

#         if identity is None:
#             identity = x_q

#         out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

#         return identity + self.dropout_layer(self.proj_drop(out))


# class PVTEncoderLayer(BaseModule):
#     """Implements one encoder layer in PVT.

#      Args:
#         embed_dims (int): The feature dimension.
#         num_heads (int): Parallel attention heads.
#         feedforward_channels (int): The hidden dimension for FFNs.
#         drop_rate (float): Probability of an element to be zeroed.
#             after the feed forward layer. Default: 0.0.
#         attn_drop_rate (float): The drop out rate for attention layer.
#             Default: 0.0.
#         drop_path_rate (float): stochastic depth rate. Default: 0.0.
#         qkv_bias (bool): enable bias for qkv if True.
#             Default: True.
#         act_cfg (dict): The activation config for FFNs.
#             Default: dict(type='GELU').
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN').
#         sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
#             Attention of PVT. Default: 1.
#         use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
#             Default: False.
#         init_cfg (dict, optional): Initialization config dict.
#             Default: None.
#     """

#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  feedforward_channels,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  qkv_bias=True,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN'),
#                  sr_ratio=1,
#                  use_conv_ffn=False,
#                  init_cfg=None):
#         super(PVTEncoderLayer, self).__init__(init_cfg=init_cfg)

#         # The ret[0] of build_norm_layer is norm name.
#         self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

#         # 替换?
#         self.att = SparseAttention(
#             shape= None, 
#             n_head=num_heads, 
#             causal=None,
#             num_local_blocks=4,
#             block=2,
#             attn_dropout=0 
#         )

#         self.attn = SpatialReductionAttention(
#             embed_dims=embed_dims,
#             num_heads=num_heads,
#             attn_drop=attn_drop_rate,
#             proj_drop=drop_rate,
#             dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#             qkv_bias=qkv_bias,
#             norm_cfg=norm_cfg,
#             sr_ratio=sr_ratio)

#         # The ret[0] of build_norm_layer is norm name.
#         self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

#         self.ffn = MixFFN(
#             embed_dims=embed_dims,
#             feedforward_channels=feedforward_channels,
#             ffn_drop=drop_rate,
#             dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#             use_conv=use_conv_ffn,
#             act_cfg=act_cfg)

#     def forward(self, x, hw_shape):
#         x = self.attn(self.norm1(x), hw_shape, identity=x)
#         # print("x.size",x.size() )
#         x = self.ffn(self.norm2(x), hw_shape, identity=x)

#         return x

# """暂时注销Abs位置嵌入方式
# """
# # class AbsolutePositionEmbedding(BaseModule):
# #     """An implementation of the absolute position embedding in PVT.

# #      Args:
# #         pos_shape (int): The shape of the absolute position embedding.
# #         pos_dim (int): The dimension of the absolute position embedding.
# #         drop_rate (float): Probability of an element to be zeroed.
# #             Default: 0.0.
# #     """

# #     def __init__(self, 
# #                             pos_shape, 
# #                             pos_dim, 
# #                             drop_rate=0., 
# #                             init_cfg=None):
# #         super().__init__(init_cfg=init_cfg)

# #         if isinstance(pos_shape, int): #检查pose_shape是否为int类型
# #             pos_shape = to_2tuple(pos_shape)
# #         elif isinstance(pos_shape, tuple):
# #             if len(pos_shape) == 1:
# #                 pos_shape = to_2tuple(pos_shape[0])
# #             assert len(pos_shape) == 2, \
# #                 f'The size of image should have length 1 or 2, ' \
# #                 f'but got {len(pos_shape)}'
# #         self.pos_shape = pos_shape
# #         self.pos_dim = pos_dim

# #         self.pos_embed = nn.Parameter(
# #             torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim)) #生成可优化的零矩阵
# #         self.drop = nn.Dropout(p=drop_rate)

# #     def init_weights(self):
# #         trunc_normal_(self.pos_embed, std=0.02)

# #     def resize_pos_embed(self, pos_embed, input_shape, mode='bilinear'):
# #         """Resize pos_embed weights.

# #         Resize pos_embed using bilinear interpolate method.

# #         Args:
# #             pos_embed (torch.Tensor): Position embedding weights.
# #             input_shape (tuple): Tuple for (downsampled input image height,
# #                 downsampled input image width).
# #             mode (str): Algorithm used for upsampling:
# #                 ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
# #                 ``'trilinear'``. Default: ``'bilinear'``.

# #         Return:
# #             torch.Tensor: The resized pos_embed of shape [B, L_new, C].
# #         """
# #         assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
# #         pos_h, pos_w = self.pos_shape
# #         pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]  #取数据
# #         pos_embed_weight = pos_embed_weight.reshape(
# #             1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous()  #变换维度 [1 h w dim]  ->  [1  dim  h  w] 
# #         pos_embed_weight = F.interpolate(
# #             pos_embed_weight, size=input_shape, mode=mode) #数组采样 mode=mode
# #         pos_embed_weight = torch.flatten(pos_embed_weight,  #第2维度展开[1 dim L] 变换 -> [1 L dim]
# #                                          2).transpose(1, 2).contiguous()
# #         pos_embed = pos_embed_weight

# #         return pos_embed

# #     def forward(self, x, hw_shape, mode='bilinear'):
# #         pos_embed = self.resize_pos_embed(self.pos_embed, hw_shape, mode)
# #         return self.drop(x + pos_embed)


# #""""
# ####################################################################################
# class SparseAttention(nn.Module):
#     ops = dict()
#     attn_mask = dict()
#     block_layout = dict()

#     def __init__(self, shape, n_head, causal, num_local_blocks=4, block=32,
#                  attn_dropout=0.): # does not use attn_dropout
#         super().__init__()
#         self.causal = causal
#         self.shape = shape

#         self.sparsity_config = StridedSparsityConfig(shape=shape, n_head=n_head,
#                                                      causal=causal, block=block,
#                                                      num_local_blocks=num_local_blocks)

#         if self.shape not in SparseAttention.block_layout:
#             SparseAttention.block_layout[self.shape] = self.sparsity_config.make_layout()
#         if causal and self.shape not in SparseAttention.attn_mask:
#             SparseAttention.attn_mask[self.shape] = self.sparsity_config.make_sparse_attn_mask()

#     def get_ops(self):
#         try:
#             from deepspeed.ops.sparse_attention import MatMul, Softmax
#         except:
#             raise Exception('Error importing deepspeed. Please install using `DS_BUILD_SPARSE_ATTN=1 pip install deepspeed`')
#         if self.shape not in SparseAttention.ops:
#             sparsity_layout = self.sparsity_config.make_layout()
#             sparse_dot_sdd_nt = MatMul(sparsity_layout,
#                                        self.sparsity_config.block,
#                                        'sdd',
#                                        trans_a=False,
#                                        trans_b=True)

#             sparse_dot_dsd_nn = MatMul(sparsity_layout,
#                                        self.sparsity_config.block,
#                                        'dsd',
#                                        trans_a=False,
#                                        trans_b=False)

#             sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)

#             SparseAttention.ops[self.shape] = (sparse_dot_sdd_nt,
#                                                sparse_dot_dsd_nn,
#                                                sparse_softmax)
#         return SparseAttention.ops[self.shape]

#     def forward(self, q, k, v, decode_step, decode_idx):
#         if self.training and self.shape not in SparseAttention.ops:
#             self.get_ops()

#         SparseAttention.block_layout[self.shape] = SparseAttention.block_layout[self.shape].to(q)
#         if self.causal:
#             SparseAttention.attn_mask[self.shape] = SparseAttention.attn_mask[self.shape].to(q).type_as(q)
#         attn_mask = SparseAttention.attn_mask[self.shape] if self.causal else None

#         old_shape = q.shape[2:-1]
#         q = q.flatten(start_dim=2, end_dim=-2)
#         k = k.flatten(start_dim=2, end_dim=-2)
#         v = v.flatten(start_dim=2, end_dim=-2)

#         if decode_step is not None:
#             mask = self.sparsity_config.get_non_block_layout_row(SparseAttention.block_layout[self.shape], decode_step)
#             out = scaled_dot_product_attention(q, k, v, mask=mask, training=self.training)
#         else:
#             if q.shape != k.shape or k.shape != v.shape:
#                 raise Exception('SparseAttention only support self-attention')
#             sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops()
#             scaling = float(q.shape[-1]) ** -0.5

#             attn_output_weights = sparse_dot_sdd_nt(q, k)
#             if attn_mask is not None:
#                 attn_output_weights = attn_output_weights.masked_fill(attn_mask == 0,
#                                                                       float('-inf'))
#             attn_output_weights = sparse_softmax(
#                 attn_output_weights,
#                 scale=scaling
#             )

#             out = sparse_dot_dsd_nn(attn_output_weights, v)

#         return view_range(out, 2, 3, old_shape)


# class StridedSparsityConfig(object):
#     """
#     Strided Sparse configuration specified in https://arxiv.org/abs/1904.10509 that
#     generalizes to arbitrary dimensions
#     """
#     def __init__(self, shape, n_head, causal, block, num_local_blocks):
#         self.n_head = n_head
#         self.shape = shape
#         self.causal = causal
#         self.block = block
#         self.num_local_blocks = num_local_blocks

#         assert self.num_local_blocks >= 1, 'Must have at least 1 local block'
#         assert self.seq_len % self.block == 0, 'seq len must be divisible by block size'

#         self._block_shape = self._compute_block_shape()
#         self._block_shape_cum = self._block_shape_cum_sizes()

#     @property
#     def seq_len(self):
#         return np.prod(self.shape)

#     @property
#     def num_blocks(self):
#         return self.seq_len // self.block

#     def set_local_layout(self, layout):
#         num_blocks = self.num_blocks
#         for row in range(0, num_blocks):
#             end = min(row + self.num_local_blocks, num_blocks)
#             for col in range(
#                     max(0, row - self.num_local_blocks),
#                     (row + 1 if self.causal else end)):
#                 layout[:, row, col] = 1
#         return layout

#     def set_global_layout(self, layout):
#         num_blocks = self.num_blocks
#         n_dim = len(self._block_shape)
#         for row in range(num_blocks):
#             assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
#             cur_idx = self._to_unflattened_idx(row)
#             # no strided attention over last dim
#             for d in range(n_dim - 1):
#                 end = self._block_shape[d]
#                 for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
#                     new_idx = list(cur_idx)
#                     new_idx[d] = i
#                     new_idx = tuple(new_idx)

#                     col = self._to_flattened_idx(new_idx)
#                     layout[:, row, col] = 1

#         return layout

#     def make_layout(self):
#         layout = torch.zeros((self.n_head, self.num_blocks, self.num_blocks), dtype=torch.int64)
#         layout = self.set_local_layout(layout)
#         layout = self.set_global_layout(layout)
#         return layout

#     def make_sparse_attn_mask(self):
#         block_layout = self.make_layout()
#         assert block_layout.shape[1] == block_layout.shape[2] == self.num_blocks

#         num_dense_blocks = block_layout.sum().item()
#         attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
#         counter = 0
#         for h in range(self.n_head):
#             for i in range(self.num_blocks):
#                 for j in range(self.num_blocks):
#                     elem = block_layout[h, i, j].item()
#                     if elem == 1:
#                         assert i >= j
#                         if i == j: # need to mask within block on diagonals
#                             attn_mask[counter] = torch.tril(attn_mask[counter])
#                         counter += 1
#         assert counter == num_dense_blocks

#         return attn_mask.unsqueeze(0)

#     def get_non_block_layout_row(self, block_layout, row):
#         block_row = row // self.block
#         block_row = block_layout[:, [block_row]] # n_head x 1 x n_blocks
#         block_row = block_row.repeat_interleave(self.block, dim=-1)
#         block_row[:, :, row + 1:] = 0.
#         return block_row

#     ###### help function #####
#     def _compute_block_shape(self):
#         n_dim = len(self.shape)
#         cum_prod = 1
#         for i in range(n_dim - 1, -1, -1):
#             cum_prod *= self.shape[i]
#             if cum_prod > self.block:
#                 break
#         assert cum_prod % self.block == 0
#         new_shape = (*self.shape[:i], cum_prod // self.block)

#         assert np.prod(new_shape) == np.prod(self.shape) // self.block

#         return new_shape

#     def _block_shape_cum_sizes(self):
#         bs = np.flip(np.array(self._block_shape))
#         return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

#     def _to_flattened_idx(self, idx):
#         assert len(idx) == len(self._block_shape), f"{len(idx)} != {len(self._block_shape)}"
#         flat_idx = 0
#         for i in range(len(self._block_shape)):
#             flat_idx += idx[i] * self._block_shape_cum[i]
#         return flat_idx

#     def _to_unflattened_idx(self, flat_idx):
#         assert flat_idx < np.prod(self._block_shape)
#         idx = []
#         for i in range(len(self._block_shape)):
#             idx.append(flat_idx // self._block_shape_cum[i])
#             flat_idx %= self._block_shape_cum[i]
#         return tuple(idx)

# """ reshapes tensor start from dim i (inclusive)
#     to dim j (exclusive) to the desired shape
#     e.g. if x.shape = (b, thw, c) then
#     view_range(x, 1, 2, (t, h, w)) returns
#     x of shape (b, t, h, w, c)"""
# def view_range(x, i, j, shape):
#     shape = tuple(shape)

#     n_dims = len(x.shape)
#     if i < 0:
#         i = n_dims + i

#     if j is None:
#         j = n_dims
#     elif j < 0:
#         j = n_dims + j

#     assert 0 <= i < j <= n_dims

#     x_shape = x.shape
#     target_shape = x_shape[:i] + shape + x_shape[j:]
#     return x.view(target_shape)


# """ tensor_slice函数
# def tensor_slice(x, begin, size):
#     assert all([b >= 0 for b in begin])
#     size = [l - b if s == -1 else s
#             for s, b, l in zip(size, begin, x.shape)]
#     assert all([s >= 0 for s in size])

#     slices = [slice(b, b + s) for b, s in zip(begin, size)]
#     return x[slices]
# """

# class AddBroadcastPosEmbed(nn.Module):
#     # def __init__(self, shape, embd_dim, dim=-1):
#     #     super().__init__()
#     #     assert dim in [-1, 1] # only first or last dim supported
#     #     self.shape = shape
#     #     self.n_dim = n_dim = len(shape)
#     #     self.embd_dim = embd_dim
#     #     self.dim = dim

#     #     assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
#     #     self.emb = nn.ParameterDict({
#     #         f'd_{i}': nn.Parameter(torch.randn(shape[i], embd_dim // n_dim) * 0.01
#     #                                 if dim == -1 else
#     #                                 torch.randn(embd_dim // n_dim, shape[i]) * 0.01)
#     #         for i in range(n_dim)
#     #     })

#     def __init__(self, pos_shape, pos_dim, dim=-1):
#         super().__init__()
#         assert dim in [-1, 1] # only first or last dim supported
#         if isinstance(pos_shape, int): #检查pose_shape是否为int类型
#             pos_shape = to_2tuple(pos_shape)
#         elif isinstance(pos_shape, tuple):
#             if len(pos_shape) == 1:
#                 pos_shape = to_2tuple(pos_shape[0])
#             assert len(pos_shape) == 2, \
#                 f'The size of image should have length 1 or 2, ' \
#                 f'but got {len(pos_shape)}'

#         self.pos_shape = pos_shape
#         self.n_dim = n_dim = len(pos_shape)
#         self.pos_dim = pos_dim
#         self.dim = dim

#         assert pos_dim % n_dim == 0, f"{pos_dim} % {n_dim} != 0"
        
#         self.pos_embed = nn.ParameterDict({
#             f'd_{i}': nn.Parameter(torch.randn(pos_shape[i], pos_dim // n_dim) * 0.01
#                                     if dim == -1 else
#                                     torch.randn(pos_dim // n_dim, pos_shape[i]) * 0.01)
#             for i in range(n_dim)
#         })

#     def forward(self, x, decode_step=None, decode_idx=None):
#         embs = []
#         for i in range(self.n_dim):
#             e = self.pos_embed[f'd_{i}']
#             if self.dim == -1:
#                 # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
#                 e = e.view(1, *((1,) * i), self.pos_shape[i], *((1,) * (self.n_dim - i - 1)), -1)
#                 e = e.expand(1, *self.pos_shape, -1)
#             else:
#                 e = e.view(1, -1, *((1,) * i), self.pos_shape[i], *((1,) * (self.n_dim - i - 1)))
#                 e = e.expand(1, -1, *self.pos_shape)
#             embs.append(e)

#         embs = torch.cat(embs, dim=self.dim)
#         # if decode_step is not None:
#         #     embs = tensor_slice(embs, [0, *decode_idx, 0],
#         #                         [x.shape[0], *(1,) * self.n_dim, x.shape[-1]])

#         return x + embs

# ###############################################

# #################Helper########################
# def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
#     # Performs scaled dot-product attention over the second to last dimension dn

#     # (b, n_head, d1, ..., dn, d)
#     attn = torch.matmul(q, k.transpose(-1, -2))
#     attn = attn / np.sqrt(q.shape[-1])
#     if mask is not None:
#         attn = attn.masked_fill(mask == 0, float('-inf'))
#     attn_float = F.softmax(attn, dim=-1)
#     attn = attn_float.type_as(attn) # b x n_head x d1 x ... x dn x d
#     attn = F.dropout(attn, p=attn_dropout, training=training)

#     a = torch.matmul(attn, v) # b x n_head x d1 x ... x dn x d

#     return a




# #"""


# class PyramidVisionTransformer_change(BaseModule):
#     """Pyramid Vision Transformer (PVT)

#         Implementation of `Pyramid Vision Transformer: A Versatile Backbone for
#         Dense Prediction without Convolutions
#         <https://arxiv.org/pdf/2102.12122.pdf>`_.

#         Args:
#             pretrain_img_size (int | tuple[int]): The size of input image when
#                 pretrain. Defaults: 224.
#             in_channels (int): Number of input channels. Default: 3.
#             embed_dims (int): Embedding dimension. Default: 64.
#             num_stags (int): The num of stages. Default: 4.
#             num_layers (Sequence[int]): The layer number of each transformer encode
#                 layer. Default: [3, 4, 6, 3].
#             num_heads (Sequence[int]): The attention heads of each transformer
#                 encode layer. Default: [1, 2, 5, 8].
#             patch_sizes (Sequence[int]): The patch_size of each patch embedding.
#                 Default: [4, 2, 2, 2].
#             strides (Sequence[int]): The stride of each patch embedding.
#                 Default: [4, 2, 2, 2].
#             paddings (Sequence[int]): The padding of each patch embedding.
#                 Default: [0, 0, 0, 0].
#             sr_ratios (Sequence[int]): The spatial reduction rate of each
#                 transformer encode layer. Default: [8, 4, 2, 1].
#             out_indices (Sequence[int] | int): Output from which stages.
#                 Default: (0, 1, 2, 3).
#             mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
#                 embedding dim of each transformer encode layer.
#                 多层感知机隐藏层的维度与输入的嵌入特征维度之比
#                 Default: [8, 8, 4, 4].
#             qkv_bias (bool): Enable bias for qkv if True. Default: True.
#             drop_rate (float): Probability of an element to be zeroed.
#                 Default 0.0.
#             attn_drop_rate (float): The drop out rate for attention layer.
#                 Default 0.0.
#             drop_path_rate (float): stochastic depth rate. Default 0.1.
#             use_abs_pos_embed (bool): If True, add absolute position embedding to
#                 the patch embedding. Defaults: True.
#             use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
#                 Default: False.
#             act_cfg (dict): The activation config for FFNs.
#                 Default: dict(type='GELU').
#             norm_cfg (dict): Config dict for normalization layer.
#                 Default: dict(type='LN').
#             pretrained (str, optional): model pretrained path. Default: None.
#             convert_weights (bool): The flag indicates whether the
#                 pre-trained model is from the original repo. We may need
#                 to convert some keys to make it compatible.
#                 Default: True.
#             init_cfg (dict or list[dict], optional): Initialization config dict.
#                 Default: None.
#     """

#     def __init__(self,
#                  pretrain_img_size=224,
#                  in_channels=3,
#                  embed_dims=64,
#                  num_stages=4,
#                  num_layers=[3, 4, 6, 3],  #用的是small版本
#                  num_heads=[1, 2, 5, 8],
#                  patch_sizes=[4, 2, 2, 2], #
#                  strides=[4, 2, 2, 2], ##注意此处
#                  paddings=[0, 0, 0, 0],#
#                  sr_ratios=[8, 4, 2, 1], #
#                  out_indices=(0, 1, 2, 3),
#                  mlp_ratios=[8, 8, 4, 4],  
#                  qkv_bias=True,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.1,
#                  use_abs_pos_embed=True,
#                  norm_after_stage=False,
#                  use_conv_ffn=False,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN', eps=1e-6),
#                  pretrained=None,
#                  convert_weights=True,
#                  init_cfg=None):
#         super().__init__(init_cfg=init_cfg)

#         self.convert_weights = convert_weights
#         if isinstance(pretrain_img_size, int):
#             pretrain_img_size = to_2tuple(pretrain_img_size)
#         elif isinstance(pretrain_img_size, tuple):
#             if len(pretrain_img_size) == 1:
#                 pretrain_img_size = to_2tuple(pretrain_img_size[0])
#             assert len(pretrain_img_size) == 2, \
#                 f'The size of image should have length 1 or 2, ' \
#                 f'but got {len(pretrain_img_size)}'

#         assert not (init_cfg and pretrained), \
#             'init_cfg and pretrained cannot be setting at the same time'
#         if isinstance(pretrained, str):
#             warnings.warn('DeprecationWarning: pretrained is deprecated, '
#                           'please use "init_cfg" instead')
#             self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
#         elif pretrained is None:
#             self.init_cfg = init_cfg
#         else:
#             raise TypeError('pretrained must be a str or None')

#         self.embed_dims = embed_dims

#         self.num_stages = num_stages
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.patch_sizes = patch_sizes
#         self.strides = strides
#         self.use_abs_pos_embed=use_abs_pos_embed
#         self.sr_ratios = sr_ratios
#         assert num_stages == len(num_layers) == len(num_heads) \
#                == len(patch_sizes) == len(strides) == len(sr_ratios)

#         self.out_indices = out_indices
#         assert max(out_indices) < self.num_stages
#         self.pretrained = pretrained

#         # transformer encoder
#         dpr = [
#             x.item()
#             for x in torch.linspace(0, drop_path_rate, sum(num_layers))  #随机层数衰减relu
#         ]  # stochastic num_layer decay rule

#         cur = 0
#         self.layers = ModuleList()  #Network
        
#         # #embed_dims_i = embed_dims * num_heads[0]
#         # self.patch_embed = PatchEmbed(
#         #         in_channels=in_channels,
#         #         #embed_dims=embed_dims_i,
#         #         embed_dims=embed_dims * num_heads[0],
#         #         kernel_size=patch_sizes[0],
#         #         stride=strides[0],
#         #         padding=paddings[0],
#         #         bias=True,
#         #         norm_cfg=norm_cfg)
#         # #self.layers.append(patch_embed)  # 0

#         # if use_abs_pos_embed:
#         #         #pos_shape = pretrain_img_size // np.prod(patch_sizes[:i + 1])
#         #         pos_shape = pretrain_img_size // np.prod(patch_sizes[:0 + 1])
#         #         self.pos_embed = AbsolutePositionEmbedding(
#         #             pos_shape=pos_shape,
#         #             #pos_dim=embed_dims_i,
#         #             pos_dim=embed_dims * num_heads[0],
#         #             drop_rate=drop_rate)

        
#         for i, num_layer in enumerate(num_layers): #enumrate 枚举索引和值 num=4
#             embed_dims_i = embed_dims * num_heads[i]
#             proj=nn.Linear(in_features=in_channels,out_features=embed_dims_i)

#             # #通过PatchEmbeding实现了图片尺寸的减小1/4-->1/2-->1/2-->1/2
#             # patch_embed = PatchEmbed(
#             #     in_channels=in_channels,
#             #     embed_dims=embed_dims_i,
#             #     kernel_size=patch_sizes[i],
#             #     stride=strides[i],
#             #     padding=paddings[i],
#             #     bias=True,
#             #     norm_cfg=norm_cfg)
#             # if use_abs_pos_embed:
#             #     pos_shape = pretrain_img_size // np.prod(patch_sizes[:i + 1])
#             #     pos_embed = AbsolutePositionEmbedding(
#             #         pos_shape=pos_shape,
#             #         pos_dim=embed_dims_i,
#             #         drop_rate=drop_rate)
#             #     layers.append(pos_embed)

#             patch_embed = PatchEmbed(
#                 in_channels=in_channels,
#                 embed_dims=embed_dims * num_heads[0],
#                 kernel_size=patch_sizes[0],
#                 stride=strides[0],
#                 padding=paddings[0],
#                 bias=True,
#                 norm_cfg=norm_cfg)
                        
#             layer_before=ModuleList()
#             layers = ModuleList()  #stage
            
#             # if use_abs_pos_embed:
#             #     #pos_shape = pretrain_img_size // np.prod(patch_sizes[:i + 1])
#             #     pos_shape = pretrain_img_size // np.prod(patch_sizes[:0 + 1])
#             #     pos_embed = AbsolutePositionEmbedding(
#             #         pos_shape=pos_shape,
#             #         #pos_dim=embed_dims_i,
#             #         pos_dim=embed_dims * num_heads[0],
#             #         drop_rate=drop_rate)
#             #     # layers.append(pos_embed)
            
#             if i<1:
#                 layer_before.append(patch_embed)
#                 if use_abs_pos_embed:
#                     pos_shape = pretrain_img_size // np.prod(patch_sizes[:0 + 1])
#                     #pos_embed way change:
#                     # pos_embed = AbsolutePositionEmbedding(
#                     #     pos_shape=pos_shape,
#                     #     pos_dim=embed_dims * num_heads[0],
#                     #     drop_rate=drop_rate)
#                     pos_embed = AddBroadcastPosEmbed(pos_shape=pos_shape,
#                         pos_dim=embed_dims,
#                         dim=-1)
#                     layer_before.append(pos_embed)
#             else :
#                 layer_before.append(proj)

#             layers.extend([
#                 PVTEncoderLayer(
#                     embed_dims=embed_dims_i,
#                     num_heads=num_heads[i],
#                     feedforward_channels=mlp_ratios[i] * embed_dims_i,
#                     drop_rate=drop_rate,
#                     attn_drop_rate=attn_drop_rate,
#                     drop_path_rate=dpr[cur + idx],
#                     qkv_bias=qkv_bias,
#                     act_cfg=act_cfg,
#                     norm_cfg=norm_cfg,
#                     sr_ratio=sr_ratios[i],
#                     use_conv_ffn=use_conv_ffn) for idx in range(num_layer) #num=num_layer[i]
#             ])
#             in_channels = embed_dims_i #下一循环的输入

#             # The ret[0] of build_norm_layer is norm name.
#             if norm_after_stage:
#                 norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
#             else:
#                 norm = nn.Identity()
#             #self.layers.append(ModuleList([patch_embed, layers, norm]))
#             self.layers.append(ModuleList([layer_before, layers, norm]))  #stage+norm
            
#             cur += num_layer  #总层数
            
        
#     def init_weights(self):
#         logger = get_root_logger()
#         if self.init_cfg is None:
#             logger.warn(f'No pre-trained weights for '
#                         f'{self.__class__.__name__}, '
#                         f'training start from scratch')
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     trunc_normal_init(m, std=.02, bias=0.)
#                 elif isinstance(m, nn.LayerNorm):
#                     constant_init(m, 1.0)
#                 elif isinstance(m, nn.Conv2d):
#                     fan_out = m.kernel_size[0] * m.kernel_size[
#                         1] * m.out_channels
#                     fan_out //= m.groups
#                     normal_init(m, 0, math.sqrt(2.0 / fan_out))
#                 #there is not AbsolutePositionEmbedding:
#                 # elif isinstance(m, AbsolutePositionEmbedding):
#                 #     m.init_weights()
#                 elif isinstance(m, AddBroadcastPosEmbed):
#                     m.init_weights()
#         else:
#             assert 'checkpoint' in self.init_cfg, f'Only support ' \
#                                                   f'specify `Pretrained` in ' \
#                                                   f'`init_cfg` in ' \
#                                                   f'{self.__class__.__name__} '
#             checkpoint = _load_checkpoint(
#                 self.init_cfg.checkpoint, logger=logger, map_location='cpu')
#             logger.warn(f'Load pre-trained model for '
#                         f'{self.__class__.__name__} from original repo')
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
#             if self.convert_weights:
#                 # Because pvt backbones are not supported by mmcls,
#                 # so we need to convert pre-trained weights to match this
#                 # implementation.
#                 state_dict = pvt_convert(state_dict)
#             load_state_dict(self, state_dict, strict=False, logger=logger)

#     def forward(self, x):
            
#         outs = []
#         outs_in=[]
#         layer_befor_j=0
#         for i, layer in enumerate(self.layers): #4个stage
#             for block in layer[0]:
#                 if i==0:
#                     if layer_befor_j<1: #patch
#                         x, hw_shape = block(x)
#                         layer_befor_j=layer_befor_j+1
#                         #print("x, hw_shape",x.size())
#                     else: #pos_embed
#                         x = block(x, hw_shape)
#                     #print("1stage-layer0")
#                 else:
#                     #print("2-4stage-layer0")
#                     x = block(x) #linear
            
#             for block in layer[1]: #layers
#                 #print("layers")
#                 x = block(x, hw_shape)

#             x = layer[2](x) #norm
#             #print("norm")


#             x_in = nlc_to_nchw(x, hw_shape)
#             if i in self.out_indices:
#                 #print("out")
#                 outs_in.append(x_in)
#                 #print(x_in.size())
#                 #print(len(outs_in))

#         return outs_in


# @BACKBONES.register_module()
# class PVT_AtrousLSHTransformer(PyramidVisionTransformer_change):
#     """Implementation of PVT_AtrousLSHTransformer"""

#     def __init__(self, **kwargs):
#         super(PVT_AtrousLSHTransformer, self).__init__(
#             #重写参数,此时是PVT v1 (tiny)
#             num_layers=[2, 2, 2, 2],  #用的是small版本
#             num_heads=[1, 1, 1, 1],  #头数
#             #patch_sizes=[4, 2, 2, 2], #Original
#             patch_sizes=[4, 4, 4, 4],
#             #strides=[4, 2, 2, 2], #Original
#             strides=[4, 4, 4, 4], 
#             paddings=[0, 0, 0, 0],#
#             #sr_ratios=[8, 4, 2, 1], #Original
#             sr_ratios=[1, 1, 1, 1],  #不进行空间缩减注意力
#             out_indices=(0, 1, 2, 3),
#             mlp_ratios=[8, 8, 4, 4],  
#             use_abs_pos_embed=True,
#             norm_after_stage=False, 
#             use_conv_ffn=False, 
#             **kwargs)

