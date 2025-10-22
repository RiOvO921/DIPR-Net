import torch
from torch import nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures

# Relative imports from the new module structure
from .modules.encoder import HybridHierarchicalEncoder
from .modules.bottleneck import HierarchicalReceptiveFieldFusion
from .modules.decoder import DFADecoder
from .modules.prototype import (
    PrototypeAwareFeatureDecoder,
    NormalizedFeatureProjector,
    _bapb_distance_map,
    _bapb_cluster,
    PrototypeSimilarityScorer
)


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

        # Encoder
        self.encoder = HybridHierarchicalEncoder(
            input_channels, n_stages, features_per_stage, conv_op,
            strides, n_ptb_per_stage, tcr_per_stage, **kwargs
        )

        # Bottleneck
        self.bottleneck_enhancer = HierarchicalReceptiveFieldFusion(
            c_s=features_per_stage[-3], c_m=features_per_stage[-2], c_l=features_per_stage[-1],
            embed_dim=mra_embedding_dim, num_classes=num_classes
        )

        # Decoders
        self.linear_decoder = DFADecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, **kwargs
        )
        # For nnU-Net compatibility
        self.decoder = self.linear_decoder

        self.prototype_decoder = PrototypeAwareFeatureDecoder(
            self.encoder, patch_size, n_conv_per_stage_decoder, **kwargs
        )

        # Prototype-related modules
        proto_decoder_out_channels = sum(self.encoder.output_channels[:-1])
        self.projection_head = NormalizedFeatureProjector(proto_decoder_out_channels, proj_channels)
        self.prototypes = nn.Parameter(torch.zeros(num_classes, 2, proj_channels), requires_grad=False)

    def forward(self, x, y=None):
        skips = self.encoder(x)

        bottleneck = self.bottleneck_enhancer((skips[-3], skips[-2], skips[-1]))
        skips_enhanced = skips[:-1] + [bottleneck]

        linear_pred_logits = self.linear_decoder(skips_enhanced)

        # Prototype branch logic
        multi_scale_features = self.prototype_decoder(skips_enhanced)
        pixel_features = self.projection_head(multi_scale_features)

        if self.training and y is not None:
            # Create masks and update prototypes
            gt_one_hot = F.one_hot(y.squeeze(1).long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            current_prototypes = torch.zeros_like(self.prototypes)
            center_masks_list, boundary_masks_list = [], []

            for cls in range(self.num_classes):
                cls_mask = gt_one_hot[:, cls:cls + 1, :, :]
                center_mask, boundary_mask = torch.zeros_like(cls_mask), torch.zeros_like(cls_mask)
                if torch.sum(cls_mask) > 0:
                    dist_map = _bapb_distance_map(cls_mask)
                    center_mask = (dist_map > self.beta).float() * cls_mask
                    boundary_mask = ((dist_map > 0) & (dist_map <= self.beta)).float() * cls_mask

                    # NOTE: Fixed a typo here. Original code used '_bpg_cluster', but definition is '_bapb_cluster'.
                    cls_batch_protos = _bapb_cluster(pixel_features, center_mask, boundary_mask)
                    current_prototypes[cls] = torch.mean(cls_batch_protos.detach(), dim=0)

                center_masks_list.append(center_mask)
                boundary_masks_list.append(boundary_mask)

            # Momentum update for global prototypes
            self.prototypes.data = self.alpha * self.prototypes.data + (1 - self.alpha) * current_prototypes

            center_masks = torch.cat(center_masks_list, dim=1)
            boundary_masks = torch.cat(boundary_masks_list, dim=1)

        proto_pred_logits = PrototypeSimilarityScorer(pixel_features, self.prototypes)

        if self.training:
            # Handle deep supervision if linear_pred_logits is a list
            if isinstance(linear_pred_logits, (list, tuple)):
                proto_pred_logits_ds = [
                    F.interpolate(proto_pred_logits, size=p.shape[-2:], mode='bilinear', align_corners=False)
                    for p in linear_pred_logits
                ]
            else:
                proto_pred_logits_ds = proto_pred_logits

            return linear_pred_logits, proto_pred_logits_ds, pixel_features, self.prototypes, center_masks, boundary_masks
        else:
            # Inference mode
            pred_l = linear_pred_logits[0] if isinstance(linear_pred_logits, (list, tuple)) else linear_pred_logits
            if proto_pred_logits.shape[-2:] != pred_l.shape[-2:]:
                proto_pred_logits = F.interpolate(proto_pred_logits, size=pred_l.shape[-2:], mode='bilinear',
                                                  align_corners=False)

            # Average softmax probabilities for final prediction
            return (F.softmax(pred_l, 1) + F.softmax(proto_pred_logits, 1)) / 2