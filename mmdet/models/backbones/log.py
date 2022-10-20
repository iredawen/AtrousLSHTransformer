# Copyright (c) OpenMMLab. All rights reserved.
from inspect import Parameter
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.utils import _pair as to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, pvt_convert


class MixFFN(BaseModule):
    """An implementation of MixFFN of PVT.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Depth-wise Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
            Default: None.
        use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
            Defaults: False.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 use_conv=False,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        if use_conv:
            # 3x3 depth wise conv to provide positional encode information
            dw_conv = Conv2d(
                in_channels=feedforward_channels,
                out_channels=feedforward_channels,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
                bias=True,
                groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, activate, drop, fc2, drop]
        if use_conv:
            layers.insert(1, dw_conv)  #insert函数,在位置1插入dw_conv
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()  #torch.nn.Identity()是一个参数不敏感的占位符

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape) #序列->图像
        out = self.layers(out)
        out = nchw_to_nlc(out) #image
        if identity is None:
            identity = x  #序列
        return identity + self.dropout_layer(out)  #序列+图像


class SpatialReductionAttention(MultiheadAttention):
    """An implementation of Spatial Reduction Attention of PVT.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            batch_first=batch_first,
            dropout_layer=dropout_layer,
            bias=qkv_bias,
            init_cfg=init_cfg)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmdet import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'SpatialReductionAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):

        x_q = x #x_q不进行空间缩减,保持不变
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape) #序列->图像
            x_kv = self.sr(x_kv) #卷积缩减
            x_kv = nchw_to_nlc(x_kv) #图像->序列
            x_kv = self.norm(x_kv) 
        else:
            x_kv = x #不进行缩减

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1) #[B,n,,D] -> [n, B, D]
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1) #恢复[n, B, D]->[B,n,,D]

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        return identity + self.dropout_layer(self.proj_drop(out))

class LogAttention(MultiheadAttention):
    def __init__(self, 
            embed_dims,  #embed_dims_i
            num_heads,  
            win_len, #H, W
            q_len, #kernel_size
            sub_len, #参数
            scale=True,
            spare=None,
            attn_drop=0, 
            proj_drop=0, 
            dropout_layer=None,  
            batch_first=True, 
            qkv_bias=True,
            norm_cfg=dict(type='LN'),
            init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            batch_first=batch_first,
            dropout_layer=dropout_layer,
            bias=qkv_bias,
            init_cfg=init_cfg
        )

        if(spare):
            print("Active log Attentiion!!!!!!!!!!!")
            mask=self.log_mask(win_len=win_len, sub_len=sub_len)
        else:
            print("GG")
            mask=torch.tril(torch.ones(win_len, sub_len)).view(1, 1, win_len, sub_len)
            
        self.register_buffer('mask_tri', mask)
        self.win_len=win_len
        self.q_len=q_len
        self.sub_len=sub_len

        # self.num_head_att=num_heads
        # #self.split_size=embed_dims*self.num_heads  #类似于embed_dims_i = embed_dims * num_heads[i]
        # self.split_size=embed_dims
        self.scale=scale
        self.query_key=nn.Conv1d(embed_dims//num_heads, embed_dims*2, self.q_len)
        self.value=Conv1D(embed_dims, 1, embed_dims//num_heads) #输出, 1, 输入
        self.c_proj=Conv1D(embed_dims//num_heads, 1, embed_dims)
        # self.attn_drop=nn.Dropout(attn_drop)
        # self.resid_drop=nn.Dropout(proj_drop)

    def log_mask(self, win_len, sub_len):
        mask=torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i]=self.row_mask(i, sub_len=sub_len, win_len=win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))  #[1, 1, win_len, win_len]
        
    def row_mask(self, index, sub_len, win_len):
        log_l=math.ceil(np.log2(sub_len))
        mask=torch.zeros((win_len), dtype=torch.float)
        if((win_len//sub_len)*2*(log_l)>index):
            mask[:(index+1)]=1
        else:
            while(index>=0):
                if((index-log_l+1)<0):
                    mask[:index]=1
                    break
                mask[index-log_l+1 : (index+1)]=1
                for i in range(0, log_l):
                    new_index=index-log_l+1-2**i
                    if((index-new_index)<=sub_len and new_index>=0):
                        mask[new_index]=1
                index -= sub_len
        return mask
        
    def attn(self, query:torch.Tensor, key, value:torch.Tensor, act_cfg=dict(type='Softmax')):
        self.act_cfg=act_cfg
        activation=build_activation_layer(act_cfg)#??
        pre_att=torch.matmul(query, key)
        if self.scale:
            pre_att=pre_att/math.sqrt(value.size(-1))
        mask=self.mask_tri[:, :, pre_att.size(-2), pre_att.size(-1)]
        pre_att=pre_att*mask+ -1e9*(1-mask)
        pre_att=activation(pre_att)
        pre_att=self.attn_drop(pre_att)
        attn=torch.matmul(pre_att, value)

        return attn

    def merge_head(self, x): 
        x= x.permute(0, 2, 1, 3).contiguous() #[B, H, N, C_h]->[B, N, H, C_h]
        new_x_shape=x.size()[:-2] + (x.size(-2)*x.size(-1), ) #[B, N, H*C_h]->[B N C]
        return x.view(*new_x_shape)
        
    def split_head(self, x, k=False): #x  [B N C]
        new_x_shape = x.size()[:-1] + (self.num_head_att, x.size(-1)//self.num_head_att) #[B, N, H, C_h]
        x=x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1) #[B, H, C_h, N]
        else:
            return x.permute(0, 2, 1, 3) #[B, H, N, C_h]

    def forword(self, x, hw_shape, identity=None):  #x = [B N C]
        if identity is None:
            identity=x
        print("attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
        value=self.value(x)  #Conv1D  x=[B, N, C_in]->[B, N, C_out] 权重学习
        qk_x=F.pad(x.permute(0, 2, 1), pad=(self.q_len -1, 0)) #对N维度左边填充q_len个0 [B, N, C]->[B, C, N]
        query_key=self.query_key(qk_x).permute(0, 2, 1) #Conv1d [B, C, N]->[B, 2C, N]->[B, N, 2C] 一维卷积
        query, key=query_key.split(self.split_size, dim=2) #[B, N, 2C]->2个[B, N, C]
        
        query=query.transpose(0, 1)
        key=key.transpose(0, 1)
        value=value.transpose(0, 1)
        out=self.attn(query=query, key=key, value=value)[0]

        out=out.transpose(0, 1)

        # query=self.split_head(query) #[B, N, C]->[B, H, N, C_h]
        # key=self.split_head(key, k=True)#[B, N, C]->[B, H, C_h, N]
        # value=self.split_head(value) #[B, N, C]->[B, H, N, C_h]
        # attn=self.attn(query, key, value) #[B, H, N, N]*[B, H, N, C_h]->[B, H, N, C_h]
        # attn=self.merge_head(attn)#[B, H, N, C_h]->[B N C]
        
        attn=self.c_proj(out)  #[B N C]->w=[C, C_ori]  [B N C_ori, ]-> [B*N, C]*[C, C_ori]->[B*N, C_ori ]->[B N C_ori]
        # return identity + self.dropout_layer(self.proj_drop(out))
        return identity + self.dropout_layer(attn)


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf=rf
        self.out_dim=out_dim
        if rf==1:
            w=torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w=torch.nn.Parameter(w)
            self.b=torch.nn.Parameter(torch.zeros(out_dim))
            print("&&&&&&&&&&&&&&&&&&&&&&&&7")
        else:
            print("111111111111111111111111111122222222222")
            raise NotImplementedError
    
    def forword(self, x): #x=[B N C] 
        if self.rf==1:
            size_out=x.size()[:-1]+(self.out_dim, ) #[B, N]->[B, N, C_out, ]
            x=torch.addmm(self.b, x.view(-1, x.size(-1)), self.w) #[B*N, C_in]X[C_in, C_out]->[B*N, C_out]
            x=x.view(*size_out) #[B, N, C_out]
            print("#################################")
        else:
            print("^^666666666666666666666666666666666")
            raise NotImplementedError
        return x


class PVTEncoderLayer(BaseModule):
    """Implements one encoder layer in PVT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default: 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 win_len, 
                 q_len, 
                 sub_len,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 use_conv_ffn=False,
                 init_cfg=None):
        super(PVTEncoderLayer, self).__init__(init_cfg=init_cfg)

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        # self.attn = SpatialReductionAttention(
        #     embed_dims=embed_dims,
        #     num_heads=num_heads,
        #     attn_drop=attn_drop_rate,
        #     proj_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     qkv_bias=qkv_bias,
        #     norm_cfg=norm_cfg,
        #     sr_ratio=sr_ratio)
        
        self.attn= LogAttention(
            embed_dims=embed_dims,  #embed_dims_i
            num_heads=num_heads,
            win_len=win_len,
            q_len=q_len, 
            sub_len=sub_len,
            scale=True,
            spare=None,
            attn_drop=attn_drop_rate, 
            proj_drop=drop_rate, 
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),  
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            use_conv=use_conv_ffn,
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        
        # x = self.attn(self.norm1(x))
        # x = self.ffn(self.norm2(x))

        return x


class AbsolutePositionEmbedding(BaseModule):
    """An implementation of the absolute position embedding in PVT.

    Args:
        pos_shape (int): The shape of the absolute position embedding.
        pos_dim (int): The dimension of the absolute position embedding.
        drop_rate (float): Probability of an element to be zeroed.
            Default: 0.0.
    """

    def __init__(self, pos_shape, pos_dim, drop_rate=0., init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pos_shape, int):
            pos_shape = to_2tuple(pos_shape) #pre_image_size/patch_size = 令牌快的个数 -> 扩展为2维
        elif isinstance(pos_shape, tuple):
            if len(pos_shape) == 1:
                pos_shape = to_2tuple(pos_shape[0])
            assert len(pos_shape) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pos_shape)}'
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim)) #pos_shape[0] * pos_shape[1] 令牌快的数量 [ 1 N C ]
        self.drop = nn.Dropout(p=drop_rate)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02) #截断正态分布,限制变量的范围

    def resize_pos_embed(self, pos_embed, input_shape, mode='bilinear'):
        """Resize pos_embed weights.

        Resize pos_embed using bilinear interpolate method.

        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'bilinear'``.

        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C].
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        print(self.pos_shape) #分快数
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):] #每一个B的第一个N所有维度C输出
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous() #[1 C h w ]
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shape, mode=mode) #向上/下采样到任意给定尺寸[1 C h w]->[1 C H W]
        pos_embed_weight = torch.flatten(pos_embed_weight,
                                         2).transpose(1, 2).contiguous() #从第二维展开[1 C H W ]->[1 C N]->[1 N C]
        pos_embed = pos_embed_weight

        return pos_embed

    def forward(self, x, hw_shape, mode='bilinear'):
        pos_embed = self.resize_pos_embed(self.pos_embed, hw_shape, mode)
        return self.drop(x + pos_embed)


class DilationConvModule(nn.Module):
    def __init__(self,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            dilation_1=1,
                            dilation_2=2,
                            dilation_3=5):
        super(DilationConvModule, self).__init__()

        self.inchannels=in_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation_1=dilation_1
        self.dilation_2=dilation_2
        self.dilation_3=dilation_3

        self.out_channels_1=in_channels*2
        self.out_channels_2=in_channels*4
        self.out_channels_3=in_channels

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels_1,
                                    kernel_size=self.kernel_size, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation_1),
            nn.BatchNorm2d(self.out_channels_1),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=self.out_channels_2,
                                    kernel_size=self.kernel_size, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation_2),
            nn.BatchNorm2d(self.out_channels_2),
            nn.ReLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=in_channels*4, out_channels=self.out_channels_3,
                                    kernel_size=self.kernel_size, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation_3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.decent_1=nn.Conv2d(in_channels=self.out_channels_1, out_channels=self.inchannels,
                                                    kernel_size=1, stride=self.stride,
                                                    padding=self.padding, dilation=1)
        self.decent_2=nn.Conv2d(in_channels=self.out_channels_2, out_channels=self.inchannels,
                                                    kernel_size=1, stride=self.stride,
                                                    padding=self.padding, dilation=1)
       
    
    def forward(self, x): 
        #Kernel size can't be greater than actual input size
        if ((x.shape[2]>((self.kernel_size-1)*self.dilation_1+1)) and (x.shape[3]>((self.kernel_size-1)*self.dilation_1+1))): 
            x=self.conv1(x)

        if ((x.shape[2]>((self.kernel_size-1)*self.dilation_2+1)) and (x.shape[3]>((self.kernel_size-1)*self.dilation_2+1))): 
            x=self.conv2(x)

        if ((x.shape[2]>((self.kernel_size-1)*self.dilation_3+1)) and (x.shape[3]>((self.kernel_size-1)*self.dilation_3+1))):
            x=self.conv3(x)
        
        if (x.shape[1]!=self.inchannels): #1x1 conv降维
            print("1*1conv decent.")
            if(x.shape[1]/self.inchannels==2):
                x=self.decent_1(x)
            elif(x.shape[1]==self.out_channels_2):
                 x=self.decent_2(x)

        print("***********************", x.shape[1])
        out_size = (x.shape[2], x.shape[3])
        return x

@BACKBONES.register_module()
class PVT_LSH_attention(BaseModule):
    """Pyramid Vision Transformer (PVT)

    Implementation of `Pyramid Vision Transformer: A Versatile Backbone for
    Dense Prediction without Convolutions
    <https://arxiv.org/pdf/2102.12122.pdf>`_.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 64.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        patch_sizes (Sequence[int]): The patch_size of each patch embedding.
            Default: [4, 2, 2, 2].
        strides (Sequence[int]): The stride of each patch embedding.
            Default: [4, 2, 2, 2].
        paddings (Sequence[int]): The padding of each patch embedding.
            Default: [0, 0, 0, 0].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer encode layer.
            Default: [8, 8, 4, 4].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: True.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 #num_layers=[3, 4, 6, 3],
                 num_layers=[2, 2, 2, 2], #tiny
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 1, 1, 1],
                 #paddings=[0, 0, 0, 0],
                 paddings=[0, 'same', 'same', 'same'],
                 #dilations=[1, 2, 5, 7],
                 #dilations=[1, 1, 1, 1],
                 sr_ratios=[8, 4, 2, 1],
                 win_len=224, 
                 q_len=8, 
                 sub_len=8,
                 #sr_ratios=[1, 1, 1, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=[8, 8, 4, 4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=True,
                 norm_after_stage=False,
                 use_conv_ffn=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 convert_weights=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.convert_weights = convert_weights
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims

        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        self.pretrained = pretrained

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            embed_dims_i = embed_dims * num_heads[i]
            #print(embed_dims_i)
            dilation=DilationConvModule(
                in_channels=in_channels
            )

            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=True,
                norm_cfg=norm_cfg)

            layers = ModuleList()
            if use_abs_pos_embed:
                pos_shape = pretrain_img_size // np.prod(patch_sizes[:i + 1])
                pos_embed = AbsolutePositionEmbedding(
                    pos_shape=pos_shape,
                    pos_dim=embed_dims_i,
                    drop_rate=drop_rate)
                layers.append(pos_embed)
            layers.extend([
                PVTEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratios[i] * embed_dims_i,
                    win_len=win_len, 
                    q_len=q_len, 
                    sub_len=sub_len,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i],
                    use_conv_ffn=use_conv_ffn) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            if norm_after_stage:
                norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            else:
                norm = nn.Identity()
            if i<=0:
                self.layers.append(ModuleList([patch_embed, layers, norm]))
            else:
                self.layers.append(ModuleList([dilation, patch_embed, layers, norm]))
            cur += num_layer

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
                elif isinstance(m, AbsolutePositionEmbedding):
                    m.init_weights()
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if self.convert_weights:
                # Because pvt backbones are not supported by mmcls,
                # so we need to convert pre-trained weights to match this
                # implementation.
                state_dict = pvt_convert(state_dict)
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            if i<=0:
                x, hw_shape = layer[0](x)  #patch
                print("patch",x.size())
                print("hw_shape", hw_shape)
                for block in layer[1]:   #layers
                    x = block(x, hw_shape)
                    print("pos,lay1,lay2",x.size())
                x = layer[2](x)  # norm

                x = nlc_to_nchw(x, hw_shape)
                if i in self.out_indices:
                    outs.append(x)
                    print(x.size())
                    print(len(outs))
            else:
                x=layer[0](x)
                x, hw_shape = layer[1](x)  #patch
                print("patch",x.size())
                print("hw_shape", hw_shape)
                for block in layer[2]:   #layers
                    x = block(x, hw_shape)
                    print("pos,lay1,lay2",x.size())
                x = layer[3](x)  # norm

                x = nlc_to_nchw(x, hw_shape)
                if i in self.out_indices:
                    outs.append(x)
                    print(x.size())
                    print(len(outs))

        return outs
