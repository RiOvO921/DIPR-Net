import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from scipy.ndimage import distance_transform_edt as distance

# Import from dynamic_network_architectures, assuming it's in the environment
from dynamic_network_architectures.architectures.plain_unet import PlainConvUNet


class PrototypeAwareFeatureDecoder(nn.Module):  # PAFD
    def __init__(self, encoder, upsample_sizes, n_conv_per_stage, **kwargs):
        super().__init__()
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        # This creates a temporary PlainConvUNet to borrow its decoder structure
        temp_unet = PlainConvUNet(1, len(encoder.output_channels), encoder.output_channels, encoder.conv_op, 3,
                                  encoder.strides, 2, 2, n_conv_per_stage, False, **kwargs)
        self.transpconvs = temp_unet.decoder.transpconvs
        self.stages = temp_unet.decoder.stages
        self.upsample_sizes = upsample_sizes

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        bottleneck = skips[-1]
        ms_features = []
        for i in range(len(self.stages)):
            skip = skips[-(i + 2)]
            transpconv = self.transpconvs[i]
            stage = self.stages[i]
            upsampled = transpconv(bottleneck)
            x = torch.cat((upsampled, skip), dim=1)
            bottleneck = stage(x)
            ms_features.append(
                F.interpolate(bottleneck, size=self.upsample_sizes, mode='bilinear', align_corners=False))
        return torch.cat(ms_features, dim=1)


class NormalizedFeatureProjector(nn.Module):  # NFP
    def __init__(self, in_channels, proj_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(proj_channels, proj_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(proj_channels, proj_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


# ===== Helper Functions & Loss =====

def _bapb_distance_map(mask: torch.Tensor):
    dist_maps = [torch.from_numpy(distance(m.cpu().numpy())).float() for m in mask.squeeze(1)]
    return torch.stack(dist_maps, dim=0).unsqueeze(1).to(mask.device)


def _bapb_cluster(features, c_mask, b_mask):
    sum_c = torch.sum(features * c_mask, dim=(2, 3))
    num_c = torch.sum(c_mask, dim=(2, 3)) + 1e-8
    proto_c = sum_c / num_c
    sum_b = torch.sum(features * b_mask, dim=(2, 3))
    num_b = torch.sum(b_mask, dim=(2, 3)) + 1e-8
    proto_b = sum_b / num_b
    return torch.stack([proto_c, proto_b], dim=1)


def PrototypeSimilarityScorer(features, prototypes):  # PSS
    B, C, H, W = features.shape
    N_CLS, _, _ = prototypes.shape
    flat_feat = features.view(B, C, H * W).permute(0, 2, 1)
    # Reshape prototypes to be (B, N_PROTOS, C) where N_PROTOS = N_CLS * 2
    flat_prot = prototypes.view(N_CLS * 2, C).t().unsqueeze(0)
    # Matmul: (B, H*W, C) @ (B, C, N_PROTOS) -> (B, H*W, N_PROTOS)
    sim = torch.matmul(flat_feat, flat_prot)
    # Reshape to (B, N_CLS, 2, H, W)
    sim = sim.permute(0, 2, 1).view(B, N_CLS, 2, H, W)
    class_sim, _ = torch.max(sim, dim=2)
    return class_sim


class PrototypeAlignmentLoss(nn.Module):  # PAL
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, features, prototypes, c_mask, b_mask, target):
        B, C, H, W = features.shape
        N_CLS, N_REGIONS, _ = prototypes.shape  # N_REGIONS is 2 (center, boundary)

        flat_feat = features.permute(0, 2, 3, 1).reshape(-1, C)
        flat_t = target.view(-1)
        flat_cm = c_mask.view(-1) > 0.5
        flat_bm = b_mask.view(-1) > 0.5

        fg_idx = torch.where((flat_cm | flat_bm) & (flat_t < N_CLS))[0]
        if len(fg_idx) == 0:
            return torch.tensor(0.0, device=features.device)

        fg_feat = flat_feat[fg_idx]
        fg_t = flat_t[fg_idx]
        fg_is_c = flat_cm[fg_idx]

        all_prot = prototypes.view(-1, C)  # Shape: (N_CLS * N_REGIONS, C)
        logits = torch.matmul(fg_feat, all_prot.t()) / self.temp

        # Positive prototype index: class_idx * 2 + (0 if center else 1)
        pos_idx = fg_t * N_REGIONS + (~fg_is_c).long()

        return self.ce(logits, pos_idx)