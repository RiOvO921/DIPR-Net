import torch
from torch import nn
from timm.models.layers import DropPath

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.helper import get_matching_pool_op, maybe_convert_scalar_to_list

# ===== Core Building Blocks =====

class _DIPR_Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x): return self.act(self.bn(self.conv(x)))


class _DIPR_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = _DIPR_Conv(c1, c_, 1, 1)
        self.cv2 = _DIPR_Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class _DIPR_LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.var(1, keepdim=True, unbiased=False).sqrt()
        return self.weight[:, None, None] * ((x - mean) / (std + self.eps)) + self.bias[:, None, None]


class SpatialChannelAttn(nn.Module):  # SCA
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv_transform = _DIPR_Conv(dim, h, 1, act=False)
        self.projection = _DIPR_Conv(dim, dim, 1, act=False)
        self.pos_encoding = _DIPR_Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv_transform(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn_scores = (q.transpose(-2, -1) @ k) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attended_vals = (v @ attn_weights.transpose(-2, -1)).view(B, C, H, W)
        output = attended_vals + self.pos_encoding(v.reshape(B, C, H, W))
        return self.projection(output)


class _ConvGatedLinearUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop_rate=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        gated_channels = int(2 * hidden_channels / 3)
        self.fc1 = nn.Conv2d(in_channels, gated_channels * 2, 1)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(gated_channels, gated_channels, 3, 1, 1, bias=True, groups=gated_channels), act_layer())
        self.fc2 = nn.Conv2d(gated_channels, out_channels, 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        shortcut = x
        x, gate = self.fc1(x).chunk(2, dim=1)
        x = self.dw_conv(x) * gate
        return shortcut + self.dropout(self.fc2(self.dropout(x)))


class ConvAttnGLU(nn.Module):  # CAGLU
    def __init__(self, channels, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = _DIPR_LayerNorm2d(channels)
        self.norm2 = _DIPR_LayerNorm2d(channels)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.ffn = _ConvGatedLinearUnit(channels)
        self.attention = SpatialChannelAttn(channels)

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class DualPathBlock(nn.Module):  # DPB
    def __init__(self, channels, tcr=0.25):
        super().__init__()
        self.trans_channels = int(channels * tcr)
        self.cnn_channels = channels - self.trans_channels
        self.cnn_branch = _DIPR_Bottleneck(self.cnn_channels, self.cnn_channels)
        self.transformer_branch = ConvAttnGLU(self.trans_channels)
        self.fusion_conv = _DIPR_Conv(channels, channels, 1, 1)

    def forward(self, x):
        cnn_feat, trans_feat = x.split((self.cnn_channels, self.trans_channels), 1)
        return self.fusion_conv(torch.cat([self.cnn_branch(cnn_feat), self.transformer_branch(trans_feat)], dim=1))


class CrossStageDualPath(nn.Module):  # CSDP
    def __init__(self, in_channels, out_channels, num_blocks=1, tcr=0.25, expansion=0.5):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        self.initial_conv = _DIPR_Conv(in_channels, 2 * mid_channels, 1, 1)
        self.final_conv = _DIPR_Conv(mid_channels * (2 + num_blocks), out_channels, 1)
        self.main_path = nn.ModuleList(DualPathBlock(mid_channels, tcr=tcr) for _ in range(num_blocks))

    def forward(self, x):
        splits = list(self.initial_conv(x).chunk(2, 1))
        splits.extend(m(splits[-1]) for m in self.main_path)
        return self.final_conv(torch.cat(splits, 1))


# ===== Top-Level Encoder =====

class HybridHierarchicalEncoder(AbstractDynamicNetworkArchitectures):  # HHE
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, strides,
                 n_ptb_per_stage, tcr_per_stage, pool='max', **kwargs):
        super().__init__()
        if isinstance(tcr_per_stage, (int, float)): tcr_per_stage = [tcr_per_stage] * n_stages
        if isinstance(n_ptb_per_stage, int): n_ptb_per_stage = [n_ptb_per_stage] * n_stages
        self.stages = nn.ModuleList()
        self.initial_conv = _DIPR_Conv(input_channels, features_per_stage[0], k=3)
        current_in = features_per_stage[0]
        for s in range(1, n_stages):
            pool_layer = get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s])
            hybrid_block = CrossStageDualPath(in_channels=current_in, out_channels=features_per_stage[s],
                                              num_blocks=n_ptb_per_stage[s], transformer_channel_ratio=tcr_per_stage[s])
            self.stages.append(nn.Sequential(pool_layer, hybrid_block))
            current_in = features_per_stage[s]
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.conv_op = conv_op
        self.conv_bias = kwargs.get('conv_bias', True)
        self.norm_op, self.norm_op_kwargs = kwargs.get('norm_op'), kwargs.get('norm_op_kwargs')
        self.nonlin, self.nonlin_kwargs = kwargs.get('nonlin'), kwargs.get('nonlin_kwargs')

    def forward(self, x):
        skips = []
        x = self.initial_conv(x)
        skips.append(x)
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips