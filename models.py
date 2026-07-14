"""
Model architectures for the AAND-augmented Teacher-Student framework.

Contains:
  - CustomStudentCNN: Lightweight CNN student (unchanged from original)
  - DINOv2Teacher: Vanilla frozen DINOv2 teacher (unchanged from original)
  - AdvancedDINOv2Teacher: DINOv2 enhanced with RAA modules (AAND Stage 1)

Reference: AAND paper (arxiv 2405.02068v2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from raa_module import ResidualAnomalyAmplification


class CustomStudentCNN(nn.Module):
    """
    Lightweight CNN student model that mimics DINOv2's patch-level features.

    Architecture maps input images to the same spatial resolution and channel
    dimension as the teacher's output (e.g., 37×37 with 1024 channels for
    dinov2_vitl14 with 518×518 input).
    """

    def __init__(self, out_channels=1024):
        super().__init__()
        # Input size is assumed to be 518x518.
        # DINOv2 with patch size 14 outputs a 37x37 feature map (518 / 14 = 37).
        # We first downsample by 14 using a large stride convolution.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=14, stride=14),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Followed by a few convolutional blocks to learn deep features
        # that can mimic the complex representations of DINOv2.
        self.blocks = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x


class DINOv2Teacher(nn.Module):
    """
    Vanilla frozen DINOv2 teacher model.

    Extracts patch-level features from the last transformer block and reshapes
    them into a spatial feature map (B, C, H, W).
    """

    def __init__(self, size='large'):
        super().__init__()
        # Using dinov2_vitl14 as requested for high accuracy
        model_name = 'dinov2_vitl14' if size == 'large' else 'dinov2_vits14'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        # Freeze teacher parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.model.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']  # Shape: (B, num_patches, embed_dim)

            # Reshape from (B, N, C) to (B, C, H, W)
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            patch_tokens = patch_tokens.transpose(1, 2).view(B, C, H, W)
            return patch_tokens


class AdvancedDINOv2Teacher(nn.Module):
    """
    DINOv2 teacher enhanced with Residual Anomaly Amplification (RAA) modules.

    The frozen DINOv2 backbone extracts features, then RAA modules at selected
    transformer blocks generate adaptive residuals that amplify anomaly signals
    while preserving normal feature integrity.

    Two modes:
      - Stage 1 training: RAA modules are trainable, gt_mask is provided.
      - Stage 2 / Inference: Entire model is frozen, gt_mask is None.

    Args:
        size: DINOv2 model size ('large' or 'small').
        num_memory_items: Number of items per memory bank in MRG.
        raa_block_indices: Which transformer blocks to attach RAA modules to.
                          Default: last block only (simplification of the paper's
                          multi-scale approach for our single-scale student).
    """

    def __init__(self, size='large', num_memory_items=50, raa_block_indices=None):
        super().__init__()

        model_name = 'dinov2_vitl14' if size == 'large' else 'dinov2_vits14'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.embed_dim = self.model.embed_dim

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Determine which blocks get RAA modules
        num_blocks = len(self.model.blocks)
        if raa_block_indices is None:
            # Default: attach to the last block
            self.raa_block_indices = [num_blocks - 1]
        else:
            self.raa_block_indices = raa_block_indices

        # Create RAA modules for selected blocks
        self.raa_modules = nn.ModuleDict()
        for idx in self.raa_block_indices:
            self.raa_modules[str(idx)] = ResidualAnomalyAmplification(
                embed_dim=self.embed_dim,
                num_memory_items=num_memory_items
            )

        print(f"AdvancedDINOv2Teacher: {model_name} with RAA at blocks {self.raa_block_indices}")
        print(f"  Backbone params (frozen): {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  RAA params (trainable):   {sum(p.numel() for p in self.raa_modules.parameters()):,}")

    def forward(self, x, gt_mask=None):
        """
        Args:
            x: (B, 3, H, W) — input images.
            gt_mask: (B, 1, H_feat, W_feat) — ground truth anomaly mask at feature
                     resolution (e.g., 37×37 for 518×518 input). Only used in Stage 1.

        Returns:
            advanced_features: (B, C, H_feat, W_feat) — enhanced features.
            anomaly_weights: (B, 1, H_feat, W_feat) — predicted anomaly weights.
                            Only meaningful if RAA is at last block; otherwise
                            returns the weights from the last RAA module.
            vanilla_features: (B, C, H_feat, W_feat) — original frozen teacher
                             features (for loss computation in Stage 1).
        """
        with torch.no_grad():
            # Run through the transformer blocks
            x_prep = self.model.prepare_tokens_with_masks(x)

            block_outputs = {}
            for i, block in enumerate(self.model.blocks):
                x_prep = block(x_prep)
                if i in self.raa_block_indices:
                    # Extract patch tokens (remove CLS)
                    patch_tokens = x_prep[:, 1:, :]  # (B, N, C)
                    B, N, C = patch_tokens.shape
                    H = W = int(N ** 0.5)
                    spatial_feat = patch_tokens.transpose(1, 2).view(B, C, H, W)
                    block_outputs[i] = spatial_feat.detach()

        # Apply RAA modules (these ARE differentiable)
        all_anomaly_weights = None
        advanced_features = None
        vanilla_features = None

        for idx in self.raa_block_indices:
            teacher_feat = block_outputs[idx]
            vanilla_features = teacher_feat  # Keep reference for loss

            adv_feat, anom_w = self.raa_modules[str(idx)](teacher_feat, gt_mask)
            advanced_features = adv_feat
            all_anomaly_weights = anom_w

        return advanced_features, all_anomaly_weights, vanilla_features

    def freeze_raa(self):
        """Freeze RAA modules for Stage 2 / inference."""
        for param in self.raa_modules.parameters():
            param.requires_grad = False
        self.raa_modules.eval()

    def unfreeze_raa(self):
        """Unfreeze RAA modules for Stage 1 training."""
        for param in self.raa_modules.parameters():
            param.requires_grad = True
        self.raa_modules.train()
