# 文件: dynamic-network-architectures/building_blocks/upsample.py

from torch import nn
from typing import Union, Sequence
from dynamic_network_architectures.building_blocks.helper import get_matching_conv

class TransposedConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 conv_op=nn.Conv2d,
                 output_padding: Union[int, Sequence[int]] = 0):
        """
        This is a wrapper for torch.nn.ConvTranspose2d and torch.nn.ConvTranspose3d.
        It is used to ensure that the output shape is correct.
        It is especially useful for anisotropic images.
        """
        super().__init__()
        # Get the correct ConvTranspose operator (2D or 3D)
        self.conv_op = get_matching_conv(conv_op, transpose=True)
        
        # Instantiate the ConvTranspose layer
        self.conv = self.conv_op(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2) if len(kernel_size) == 2 else (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2) if len(kernel_size) == 3 else kernel_size // 2,
            output_padding=output_padding,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)