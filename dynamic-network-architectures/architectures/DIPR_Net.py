import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Type, Union
from scipy.ndimage import distance_transform_edt as distance
from timm.models.layers import DropPath

try:
    from mmengine.model.utils import kaiming_init, constant_init
except ImportError:
    try:
        from mmcv.cnn import kaiming_init, constant_init
    except ImportError:
        raise ImportError("Please install either mmcv or mmengine for weight initialization.")

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.helper import (
    get_matching_pool_op, maybe_convert_scalar_to_list, get_matching_convtransp
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.architectures.plain_unet import PlainConvUNet

# =================================================================================================
# ===== PART 23: DEFINITION OF ALL REQUIRED MODULES WITH CUSTOM NAMES =====
# =================================================================================================

# ----------------------------------------------------------
# Section 23.23: Core Building Blocks for HybridHierarchicalEncoder (HHE)
# ----------------------------------------------------------

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
        c_ = int(c2 * e);
        self.cv1 = _DIPR_Conv(c1, c_, 1, 1);
        self.cv2 = _DIPR_Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class _DIPR_LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__();
        self.weight = nn.Parameter(torch.ones(num_channels));
        self.bias = nn.Parameter(torch.zeros(num_channels));
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True);
        std = x.var(1, keepdim=True, unbiased=False).sqrt()
        return self.weight[:, None, None] * ((x - mean) / (std + self.eps)) + self.bias[:, None, None]


class SpatialChannelAttn(nn.Module):  # SCA
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__();
        self.num_heads = num_heads;
        self.head_dim = dim // num_heads;
        self.key_dim = int(self.head_dim * attn_ratio);
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads;
        h = dim + nh_kd * 2;
        self.qkv_transform = _DIPR_Conv(dim, h, 1, act=False);
        self.projection = _DIPR_Conv(dim, dim, 1, act=False)
        self.pos_encoding = _DIPR_Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape;
        N = H * W;
        qkv = self.qkv_transform(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn_scores = (q.transpose(-2, -1) @ k) * self.scale;
        attn_weights = attn_scores.softmax(dim=-1)
        attended_vals = (v @ attn_weights.transpose(-2, -1)).view(B, C, H, W)
        output = attended_vals + self.pos_encoding(v.reshape(B, C, H, W));
        return self.projection(output)


class _ConvGatedLinearUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop_rate=0.):
        super().__init__();
        out_channels = out_channels or in_channels;
        hidden_channels = hidden_channels or in_channels;
        gated_channels = int(2 * hidden_channels / 3)
        self.fc1 = nn.Conv2d(in_channels, gated_channels * 2, 1);
        self.dw_conv = nn.Sequential(
            nn.Conv2d(gated_channels, gated_channels, 3, 1, 1, bias=True, groups=gated_channels), act_layer())
        self.fc2 = nn.Conv2d(gated_channels, out_channels, 1);
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        shortcut = x;
        x, gate = self.fc1(x).chunk(2, dim=1);
        x = self.dw_conv(x) * gate
        return shortcut + self.dropout(self.fc2(self.dropout(x)))


class ConvAttnGLU(nn.Module):  # CAGLU
    def __init__(self, channels, drop_path_rate=0.1):
        super().__init__();
        self.norm1 = _DIPR_LayerNorm2d(channels);
        self.norm2 = _DIPR_LayerNorm2d(channels)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.ffn = _ConvGatedLinearUnit(channels);
        self.attention = SpatialChannelAttn(channels)

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)));
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class DualPathBlock(nn.Module):  # DPB
    def __init__(self, channels, tcr=0.25):
        super().__init__();
        self.trans_channels = int(channels * tcr);
        self.cnn_channels = channels - self.trans_channels
        self.cnn_branch = _DIPR_Bottleneck(self.cnn_channels, self.cnn_channels)
        self.transformer_branch = ConvAttnGLU(self.trans_channels);
        self.fusion_conv = _DIPR_Conv(channels, channels, 1, 1)

    def forward(self, x):
        cnn_feat, trans_feat = x.split((self.cnn_channels, self.trans_channels), 1)
        return self.fusion_conv(torch.cat([self.cnn_branch(cnn_feat), self.transformer_branch(trans_feat)], dim=1))


class CrossStageDualPath(nn.Module):  # CSDP
    def __init__(self, in_channels, out_channels, num_blocks=1, tcr=0.25, expansion=0.5):
        super().__init__();
        mid_channels = int(out_channels * expansion)
        self.initial_conv = _DIPR_Conv(in_channels, 2 * mid_channels, 1, 1)
        self.final_conv = _DIPR_Conv(mid_channels * (2 + num_blocks), out_channels, 1)
        self.main_path = nn.ModuleList(DualPathBlock(mid_channels, tcr=tcr) for _ in range(num_blocks))

    def forward(self, x):
        splits = list(self.initial_conv(x).chunk(2, 1))
        splits.extend(m(splits[-1]) for m in self.main_path)
        return self.final_conv(torch.cat(splits, 1))


# ----------------------------------------------------------
# Section 23.2: HierarchicalReceptiveFieldFusion (HRFF) and BoundarySensitiveAttention (BSA)
# ----------------------------------------------------------
class BoundarySensitiveAttention(nn.Module):  # BSA
    def __init__(self, channels, num_classes=2):
        super().__init__();
        self.sigmoid = nn.Sigmoid()
        self.enhancement = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.GELU())
        self.cls1 = nn.Conv2d(channels, num_classes, 1);
        self.cls2 = nn.Conv2d(channels, num_classes, 1)
        self.region_learner = nn.Sequential(nn.Conv2d(num_classes, 1, 1), nn.BatchNorm2d(1), nn.GELU())

    def forward(self, x1, x2):
        prob1 = torch.sigmoid(self.cls1(x1));
        prob2 = torch.sigmoid(self.cls2(x2))
        diff_map = torch.abs(prob1 - prob2);
        attn = self.sigmoid(self.region_learner(diff_map))
        return self.enhancement(attn * x1 + x1)


class HierarchicalReceptiveFieldFusion(nn.Module):  # HRFF
    def __init__(self, c_s, c_m, c_l, embed_dim, num_classes=2, drop_rate=0.2):
        super().__init__();
        self.conv_s = nn.Conv2d(c_s, embed_dim, 1);
        self.pool_s = nn.MaxPool2d(4)
        self.conv_m = nn.Conv2d(c_m, embed_dim, 1);
        self.pool_m = nn.MaxPool2d(2);
        self.conv_l = nn.Conv2d(c_l, embed_dim, 1)
        self.fusion = nn.Sequential(nn.Conv2d(embed_dim * 3, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.GELU())
        self.dropout = nn.Dropout2d(drop_rate)
        self.bsa1 = BoundarySensitiveAttention(embed_dim, num_classes)
        self.bsa2 = BoundarySensitiveAttention(embed_dim, num_classes)

    def forward(self, features):
        x_s, x_m, x_l = features
        x_s_p = self.pool_s(self.conv_s(x_s));
        x_m_p = self.pool_m(self.conv_m(x_m));
        x_l_p = self.conv_l(x_l)
        fused = self.fusion(torch.cat([x_s_p, x_m_p, x_l_p], dim=1))
        edge1 = self.bsa1(x_m_p, x_s_p);
        edge2 = self.bsa2(x_l_p, x_m_p)
        return self.dropout(fused) + edge1 + edge2


# ----------------------------------------------------------
# Section 23.3: DualFocusAttention (DFA) and its Dependencies
# ----------------------------------------------------------
class _DFA_ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(_DFA_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class _DFA_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(_DFA_SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DualFocusAttention(nn.Module):  # DFA
    def __init__(self, channels):
        super().__init__();
        self.ca = _DFA_ChannelAttention(channels);
        self.sa = _DFA_SpatialAttention()

    def forward(self, x):
        x_local = x
        if x.dim() == 4 and x.size(2) > 1 and x.size(3) > 1:
            top, bottom = x.chunk(2, dim=2);
            tl, tr = top.chunk(2, dim=3);
            bl, br = bottom.chunk(2, dim=3)
            quads = [tl, tr, bl, br];
            proc_quads = [self.sa(self.ca(q) * q) * q for q in quads]
            top_p = torch.cat((proc_quads[0], proc_quads[1]), dim=3);
            bot_p = torch.cat((proc_quads[2], proc_quads[3]), dim=3)
            x_local = torch.cat((top_p, bot_p), dim=2)
        x_global = self.sa(self.ca(x) * x) * x;
        return x_local + x_global


# ----------------------------------------------------------
# Section 23.4: BoundaryAwarePrototypeBank (BAPB) Framework Auxiliary Modules
# ----------------------------------------------------------
class PrototypeAwareFeatureDecoder(nn.Module):  # PAFD
    def __init__(self, encoder, upsample_sizes, n_conv_per_stage, **kwargs):
        super().__init__();
        self.transpconvs = nn.ModuleList();
        self.stages = nn.ModuleList()
        temp_unet = PlainConvUNet(1, len(encoder.output_channels), encoder.output_channels, encoder.conv_op, 3,
                                  encoder.strides, 2, 2, n_conv_per_stage, False, **kwargs)
        self.transpconvs = temp_unet.decoder.transpconvs;
        self.stages = temp_unet.decoder.stages;
        self.upsample_sizes = upsample_sizes

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        bottleneck = skips[-1];
        ms_features = []
        for i in range(len(self.stages)):
            skip = skips[-(i + 2)];
            transpconv = self.transpconvs[i];
            stage = self.stages[i]
            upsampled = transpconv(bottleneck);
            x = torch.cat((upsampled, skip), dim=1);
            bottleneck = stage(x)
            ms_features.append(
                F.interpolate(bottleneck, size=self.upsample_sizes, mode='bilinear', align_corners=False))
        return torch.cat(ms_features, dim=1)


class NormalizedFeatureProjector(nn.Module):  # NFP
    def __init__(self, in_channels, proj_channels=256):
        super().__init__();
        self.proj = nn.Sequential(nn.Conv2d(in_channels, proj_channels, 1), nn.ReLU(True),
                                  nn.Conv2d(proj_channels, proj_channels, 1), nn.ReLU(True),
                                  nn.Conv2d(proj_channels, proj_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor: return F.normalize(self.proj(x), p=2, dim=1)


def _bapb_distance_map(mask: torch.Tensor):
    dist_maps = [torch.from_numpy(distance(m.cpu().numpy())).float() for m in mask.squeeze(1)]
    return torch.stack(dist_maps, dim=0).unsqueeze(1).to(mask.device)


def _bapb_cluster(features, c_mask, b_mask):
    sum_c = torch.sum(features * c_mask, dim=(2, 3));
    num_c = torch.sum(c_mask, dim=(2, 3)) + 1e-8
    proto_c = sum_c / num_c
    sum_b = torch.sum(features * b_mask, dim=(2, 3));
    num_b = torch.sum(b_mask, dim=(2, 3)) + 1e-8
    proto_b = sum_b / num_b
    return torch.stack([proto_c, proto_b], dim=1)


def PrototypeSimilarityScorer(features, prototypes):  # PSS
    B, C, H, W = features.shape;
    N_CLS, _, _ = prototypes.shape
    flat_feat = features.view(B, C, H * W).permute(0, 2, 1)
    flat_prot = prototypes.view(-1, C).t().unsqueeze(0)
    sim = torch.matmul(flat_feat, flat_prot).permute(0, 2, 1).view(B, N_CLS, 2, H, W)
    class_sim, _ = torch.max(sim, dim=2);
    return class_sim


class PrototypeAlignmentLoss(nn.Module):  # PAL
    def __init__(self, temperature=0.1): super().__init__(); self.temp = temperature; self.ce = nn.CrossEntropyLoss()

    def forward(self, features, prototypes, c_mask, b_mask, target):
        B, C, H, W = features.shape;
        N_CLS, _, _ = prototypes.shape
        flat_feat = features.permute(0, 2, 3, 1).reshape(-1, C);
        flat_t = target.view(-1)
        flat_cm = c_mask.view(-1) > 0.5;
        flat_bm = b_mask.view(-1) > 0.5
        fg_idx = torch.where((flat_cm | flat_bm) & (flat_t < N_CLS))[0]
        if len(fg_idx) == 0: return torch.tensor(0.0, device=features.device)
        fg_feat = flat_feat[fg_idx];
        fg_t = flat_t[fg_idx];
        fg_is_c = flat_cm[fg_idx]
        all_prot = prototypes.view(-1, C);
        logits = torch.matmul(fg_feat, all_prot.t()) / self.temp
        pos_idx = fg_t * 2 + (~fg_is_c).long();
        return self.ce(logits, pos_idx)


# =================================================================================================
# ===== PART 2: DEFINITION OF RENAMED ENCODER AND DECODER =====
# =================================================================================================

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
        self.output_channels = features_per_stage;
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.conv_op = conv_op;
        self.conv_bias = kwargs.get('conv_bias', True)
        self.norm_op, self.norm_op_kwargs = kwargs.get('norm_op'), kwargs.get('norm_op_kwargs')
        self.nonlin, self.nonlin_kwargs = kwargs.get('nonlin'), kwargs.get('nonlin_kwargs')

    def forward(self, x):
        skips = [];
        x = self.initial_conv(x);
        skips.append(x)
        for stage in self.stages: x = stage(x); skips.append(x)
        return skips


class DFADecoder(nn.Module):  # Dual-Focus Attention Decoder
    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        self.stages = nn.ModuleList();
        self.transpconvs = nn.ModuleList();
        self.seg_layers = nn.ModuleList()
        self.regional_attn_modules = nn.ModuleList()  # Formerly LPA Modules
        for i in range(len(encoder.output_channels) - 1):
            skip_ch = encoder.output_channels[-(i + 2)];
            deeper_ch = encoder.output_channels[-(i + 1)]
            self.regional_attn_modules.append(DualFocusAttention(channels=skip_ch))
            self.transpconvs.append(transpconv_op(deeper_ch, skip_ch, kernel_size=encoder.strides[-(i + 1)],
                                                  stride=encoder.strides[-(i + 1)], bias=encoder.conv_bias))
            self.stages.append(StackedConvBlocks(n_conv_per_stage, skip_ch + skip_ch, skip_ch, 3, 1, encoder.conv_bias,
                                                 encoder.norm_op, encoder.norm_op_kwargs, None, None, encoder.nonlin,
                                                 encoder.nonlin_kwargs))
            if deep_supervision: self.seg_layers.append(encoder.conv_op(skip_ch, num_classes, 1, 1, 0, bias=True))
        if not deep_supervision: self.seg_layers.append(
            encoder.conv_op(encoder.output_channels[0], num_classes, 1, 1, 0, bias=True))

    def forward(self, skips):
        bottleneck = skips[-1];
        outputs = []
        for i in range(len(self.stages)):
            skip = self.regional_attn_modules[i](skips[-(i + 2)])
            transpconv = self.transpconvs[i];
            stage = self.stages[i]
            upsampled = transpconv(bottleneck);
            x = torch.cat((upsampled, skip), dim=1)
            bottleneck = stage(x)
            if self.deep_supervision: outputs.append(self.seg_layers[i](bottleneck))
        if not self.deep_supervision: return self.seg_layers[0](bottleneck)
        return outputs[::-1]


# =================================================================================================
# ===== PART 3: DEFINITION OF THE FINAL DIPR_Net MODEL =====
# =================================================================================================

class DIPR_Net(AbstractDynamicNetworkArchitectures):
    def __init__(self,
                 input_channels, n_stages, features_per_stage, conv_op, strides,
                 n_ptb_per_stage, tcr_per_stage,
                 n_conv_per_stage_decoder,
                 mra_embedding_dim,
                 num_classes, deep_supervision, patch_size,
                 beta, alpha, proj_channels,
                 **kwargs):
        super().__init__()
        self.num_classes, self.beta, self.alpha = num_classes, beta, alpha

        self.encoder = HybridHierarchicalEncoder(input_channels, n_stages, features_per_stage, conv_op,
                                                 strides, n_ptb_per_stage, tcr_per_stage, **kwargs)

        self.linear_decoder = DFADecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                         **kwargs)
        self.decoder = self.linear_decoder

        self.prototype_decoder = PrototypeAwareFeatureDecoder(self.encoder, patch_size, n_conv_per_stage_decoder,
                                                              **kwargs)

        self.bottleneck_enhancer = HierarchicalReceptiveFieldFusion(
            c_s=features_per_stage[-3], c_m=features_per_stage[-2], c_l=features_per_stage[-1],
            embed_dim=mra_embedding_dim, num_classes=num_classes
        )

        proto_decoder_out_channels = sum(self.encoder.output_channels[:-1])
        self.projection_head = NormalizedFeatureProjector(proto_decoder_out_channels, proj_channels)

        # This is the BoundaryAwarePrototypeBank (BAPB)
        self.prototypes = nn.Parameter(torch.zeros(num_classes, 2, proj_channels), requires_grad=False)

    def forward(self, x, y=None):
        skips = self.encoder(x)

        bottleneck = self.bottleneck_enhancer((skips[-3], skips[-2], skips[-1]))
        skips_enhanced = skips[:-1] + [bottleneck]

        linear_pred_logits = self.linear_decoder(skips_enhanced)

        multi_scale_features = self.prototype_decoder(skips_enhanced)
        pixel_features = self.projection_head(multi_scale_features)

        if self.training and y is not None:
            gt_one_hot = F.one_hot(y.squeeze(1).long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            current_prototypes = torch.zeros_like(self.prototypes)
            center_masks_list, boundary_masks_list = [], []
            for cls in range(self.num_classes):
                cls_mask = gt_one_hot[:, cls:cls + 1, :, :]
                center_mask, boundary_mask = torch.zeros_like(cls_mask), torch.zeros_like(cls_mask)
                if torch.sum(cls_mask) > 0:
                    dist_map = _bapb_distance_map(cls_mask)
                    center_mask = (dist_map > self.beta).float()
                    boundary_mask = ((dist_map > 0) & (dist_map <= self.beta)).float()
                    cls_batch_protos = _bpg_cluster(pixel_features, center_mask, boundary_mask)
                    # Detach to prevent gradients flowing through batch statistics to global prototypes
                    current_prototypes[cls] = torch.mean(cls_batch_protos.detach(), dim=0)
                center_masks_list.append(center_mask);
                boundary_masks_list.append(boundary_mask)
            self.prototypes.data = self.alpha * self.prototypes.data + (1 - self.alpha) * current_prototypes
            center_masks = torch.cat(center_masks_list, dim=1);
            boundary_masks = torch.cat(boundary_masks_list, dim=1)

        proto_pred_logits = PrototypeSimilarityScorer(pixel_features, self.prototypes)

        if self.training:
            if isinstance(linear_pred_logits, (list, tuple)):
                proto_pred_logits = [
                    F.interpolate(proto_pred_logits, size=p.shape[-2:], mode='bilinear', align_corners=False) for p in
                    linear_pred_logits]
            return linear_pred_logits, proto_pred_logits, pixel_features, self.prototypes, center_masks, boundary_masks
        else:
            pred_l = linear_pred_logits[0] if isinstance(linear_pred_logits, list) else linear_pred_logits
            if proto_pred_logits.shape[-2:] != pred_l.shape[-2:]:
                proto_pred_logits = F.interpolate(proto_pred_logits, size=pred_l.shape[-2:], mode='bilinear',
                                                  align_corners=False)
            return (F.softmax(pred_l, 1) + F.softmax(proto_pred_logits, 1)) / 2