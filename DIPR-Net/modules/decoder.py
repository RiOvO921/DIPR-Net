import torch
from torch import nn

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


# ===== Core Building Blocks =====

class _DFA_ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.ca = _DFA_ChannelAttention(channels)
        self.sa = _DFA_SpatialAttention()

    def forward(self, x):
        x_local = x
        if x.dim() == 4 and x.size(2) > 1 and x.size(3) > 1:
            top, bottom = x.chunk(2, dim=2)
            tl, tr = top.chunk(2, dim=3)
            bl, br = bottom.chunk(2, dim=3)
            quads = [tl, tr, bl, br]
            proc_quads = [self.sa(self.ca(q) * q) * q for q in quads]
            top_p = torch.cat((proc_quads[0], proc_quads[1]), dim=3)
            bot_p = torch.cat((proc_quads[2], proc_quads[3]), dim=3)
            x_local = torch.cat((top_p, bot_p), dim=2)
        x_global = self.sa(self.ca(x) * x) * x
        return x_local + x_global


# ===== Top-Level Decoder =====

class DFADecoder(nn.Module):  # Dual-Focus Attention Decoder
    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        self.stages = nn.ModuleList()
        self.transpconvs = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.regional_attn_modules = nn.ModuleList()

        for i in range(len(encoder.output_channels) - 1):
            skip_ch = encoder.output_channels[-(i + 2)]
            deeper_ch = encoder.output_channels[-(i + 1)]
            self.regional_attn_modules.append(DualFocusAttention(channels=skip_ch))
            self.transpconvs.append(transpconv_op(deeper_ch, skip_ch, kernel_size=encoder.strides[-(i + 1)],
                                                  stride=encoder.strides[-(i + 1)], bias=encoder.conv_bias))
            self.stages.append(StackedConvBlocks(n_conv_per_stage, skip_ch * 2, skip_ch, 3, 1, encoder.conv_bias,
                                                 encoder.norm_op, encoder.norm_op_kwargs, None, None, encoder.nonlin,
                                                 encoder.nonlin_kwargs))
            if deep_supervision: self.seg_layers.append(encoder.conv_op(skip_ch, num_classes, 1, 1, 0, bias=True))

        if not deep_supervision:
            self.seg_layers.append(encoder.conv_op(encoder.output_channels[0], num_classes, 1, 1, 0, bias=True))

    def forward(self, skips):
        bottleneck = skips[-1]
        outputs = []
        for i in range(len(self.stages)):
            skip = self.regional_attn_modules[i](skips[-(i + 2)])
            transpconv = self.transpconvs[i]
            stage = self.stages[i]
            upsampled = transpconv(bottleneck)
            x = torch.cat((upsampled, skip), dim=1)
            bottleneck = stage(x)
            if self.deep_supervision:
                outputs.append(self.seg_layers[i](bottleneck))

        if not self.deep_supervision:
            return self.seg_layers[0](bottleneck)

        return outputs[::-1]