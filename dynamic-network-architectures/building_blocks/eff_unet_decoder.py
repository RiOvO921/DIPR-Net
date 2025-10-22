# 文件: dynamic_network_architectures/building_blocks/eff_unet_decoder.py

from torch import nn
import torch
from typing import List, Union, Tuple

# --- 核心修复点：只导入确定存在的函数 ---
# 既然 UNetDecoder.py 能导入它，我们也应该能
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, List, Tuple

# --- SpatialAttention (EFF 的依赖) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
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

# --- EfficientChannelAttention (EFF 的依赖) ---
class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return v

# --- Efficient_Attention_Gate (EFF 的依赖) ---
class Efficient_Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Efficient_Attention_Gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # 返回加权后的 skip connection
        return x * psi

# --- EFF 模块本身 ---
class EFF(nn.Module):
    def __init__(self, in_dim, out_dim, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        
        # 在瓶颈处，没有 skip connection，所以 EAG 被禁用
        if not is_bottom:
            # g (x) 的通道数是 out_dim, l (skip) 的通道数是 in_dim
            # 注意通道数的匹配
            self.EAG = Efficient_Attention_Gate(F_g=out_dim, F_l=in_dim, F_int=in_dim)
        else:
            self.EAG = nn.Identity()
            
        # 拼接后的通道数。如果非瓶颈，是 out_dim (来自x) + in_dim (来自EAG_skip)
        # 如果是瓶颈，只有 in_dim (就是x自己)
        concat_channels = out_dim + in_dim if not is_bottom else in_dim
        
        self.ECA = EfficientChannelAttention(concat_channels)
        self.SA = SpatialAttention()
        
        # 最后用一个卷积来调整通道数到 out_dim
        self.final_conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        # x 是来自深层的上采样特征
        # skip 是来自编码器的跳跃连接
        
        if not self.is_bottom:
            # 23. 使用 EAG 智能过滤 skip connection
            EAG_skip = self.EAG(g=x, x=skip)
            # 2. 将过滤后的 skip 与 深层特征 x 拼接
            x = torch.cat((EAG_skip, x), dim=1)
        else:
            # 在瓶颈处，x 就是自身，没有 skip
            x = self.EAG(x)
            
        # 3. 对拼接后的特征应用 ECA (通道注意力)
        x = self.ECA(x) * x
        # 4. 接着应用 SA (空间注意力)
        x = self.SA(x) * x
        
        # 5. 最后通过卷积调整维度，作为本阶段的输出
        x = self.final_conv(x)
        return x


class EFF_UNetDecoder(nn.Module):
    def __init__(self, encoder, num_classes, deep_supervision=False):
        super().__init__()
        # ... (前半部分 __init__ 不变) ...
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes

        n_stages_encoder = len(encoder.output_channels)
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        
        # --- 核心修复点：不再导入 get_matching_conv，直接复用 encoder.conv_op ---
        conv_op = encoder.conv_op

        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        
        for i in range(n_stages_encoder - 1):
            deeper_ch = encoder.output_channels[-(i + 1)]
            skip_ch = encoder.output_channels[-(i + 2)]
            
            upsample_layer = transpconv_op(
                deeper_ch, skip_ch, 
                kernel_size=encoder.strides[-(i + 1)], 
                stride=encoder.strides[-(i + 1)], 
                bias=encoder.conv_bias
            )
            eff_block = EFF(in_dim=skip_ch, out_dim=skip_ch)
            self.stages.append(nn.ModuleList([upsample_layer, eff_block]))

            if deep_supervision:
                # 使用复用的 conv_op
                self.seg_layers.append(
                    conv_op(skip_ch, num_classes, 1, 1, 0, bias=True)
                )
        
        if not deep_supervision:
            # 使用复用的 conv_op
            self.seg_layers.append(
                conv_op(encoder.output_channels[0], num_classes, 1, 1, 0, bias=True)
            )

    # ... (forward 方法保持不变)
    def forward(self, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        l_bottleneck = skips[-1]
        outputs = []
        for i in range(len(self.stages)):
            skip_connection = skips[-(i + 2)]
            upsample_layer, eff_block = self.stages[i]
            l_upsampled = upsample_layer(l_bottleneck)
            l_bottleneck = eff_block(x=l_upsampled, skip=skip_connection)
            if self.deep_supervision:
                outputs.append(self.seg_layers[i](l_bottleneck))
        if not self.deep_supervision:
            return self.seg_layers[0](l_bottleneck)
        else:
            return outputs[::-1]