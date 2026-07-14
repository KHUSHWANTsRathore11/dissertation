"""
Residual Anomaly Amplification (RAA) Module for AAND Framework.

Contains two sub-modules:
  1. Matching-guided Residual Gate (MRG): Controls the proportion of residual
     using dual memory banks (normal + anomaly).
  2. Attribute-scaling Residual Generator (ARG): Generates channel-wise
     adaptive residuals via learned scaling weights.

Combined: Advanced_Feature = Teacher_Feature + w_a * Residual

Reference: AAND paper (arxiv 2405.02068v2), Section III-B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MatchingGuidedResidualGate(nn.Module):
    """
    Matching-guided Residual Gate (MRG).

    Maintains dual learnable memory banks (normal + anomaly). For each input
    patch feature, computes cosine similarity against both memory banks and
    outputs an anomaly weight w_a ∈ [0, 1] that controls residual proportion.

    Args:
        embed_dim: Feature dimension (C) from the teacher model.
        num_memory_items: Number of learnable embeddings (L) per memory bank.
    """

    def __init__(self, embed_dim, num_memory_items=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_memory_items = num_memory_items

        # Projection MLP: project teacher features to query space
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # Dual memory banks — learnable embeddings initialized from N(0, 1)
        self.memory_normal = nn.Parameter(torch.randn(num_memory_items, embed_dim))
        self.memory_anomaly = nn.Parameter(torch.randn(num_memory_items, embed_dim))

        # Positional encoding (learnable)
        # Will be resized at forward time based on spatial dimensions
        self.pos_embed = None

    def _get_pos_encoding(self, H, W, device):
        """Generate or cache sinusoidal 2D positional encoding."""
        if self.pos_embed is not None and self.pos_embed.shape[1] == H * W:
            return self.pos_embed.to(device)

        # Generate sinusoidal 2D positional encoding
        pos_h = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)
        pos_w = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0)

        dim = self.embed_dim
        half_dim = dim // 4

        # Frequency bands
        freq = torch.exp(torch.arange(0, half_dim, dtype=torch.float32, device=device) *
                         -(math.log(10000.0) / half_dim))

        # Encode positions
        pe_h_sin = torch.sin(pos_h * freq.unsqueeze(0))  # (H, half_dim)
        pe_h_cos = torch.cos(pos_h * freq.unsqueeze(0))
        pe_w_sin = torch.sin(pos_w.T * freq.unsqueeze(0))  # (W, half_dim)
        pe_w_cos = torch.cos(pos_w.T * freq.unsqueeze(0))

        # Combine: (H, W, dim)
        pe = torch.zeros(H, W, dim, device=device)
        pe[:, :, :half_dim] = pe_h_sin.unsqueeze(1).expand(-1, W, -1)
        pe[:, :, half_dim:2*half_dim] = pe_h_cos.unsqueeze(1).expand(-1, W, -1)
        pe[:, :, 2*half_dim:3*half_dim] = pe_w_sin.unsqueeze(0).expand(H, -1, -1)
        pe[:, :, 3*half_dim:4*half_dim] = pe_w_cos.unsqueeze(0).expand(H, -1, -1)

        self.pos_embed = pe.reshape(1, H * W, dim)
        return self.pos_embed

    def forward(self, teacher_features):
        """
        Args:
            teacher_features: (B, C, H, W) — features from frozen teacher block.

        Returns:
            anomaly_weights: (B, 1, H, W) — per-patch anomaly probability.
        """
        B, C, H, W = teacher_features.shape

        # Reshape to (B, H*W, C)
        x = teacher_features.flatten(2).transpose(1, 2)  # (B, N, C)

        # Add positional encoding
        pos = self._get_pos_encoding(H, W, x.device)
        x = x + pos

        # Project queries
        queries = self.projection(x)  # (B, N, C)

        # Normalize queries and memory items for cosine similarity
        queries_norm = F.normalize(queries, dim=-1)  # (B, N, C)
        mem_n_norm = F.normalize(self.memory_normal, dim=-1)  # (L, C)
        mem_a_norm = F.normalize(self.memory_anomaly, dim=-1)  # (L, C)

        # Cosine similarity: (B, N, L)
        sim_normal = torch.matmul(queries_norm, mem_n_norm.T)    # (B, N, L)
        sim_anomaly = torch.matmul(queries_norm, mem_a_norm.T)   # (B, N, L)

        # Concatenate and softmax over all 2L items
        sim_all = torch.cat([sim_normal, sim_anomaly], dim=-1)   # (B, N, 2L)
        weights_all = F.softmax(sim_all, dim=-1)                 # (B, N, 2L)

        # Sum of anomaly weights = anomaly probability
        anomaly_weights = weights_all[:, :, self.num_memory_items:].sum(dim=-1)  # (B, N)

        # Reshape to (B, 1, H, W)
        anomaly_weights = anomaly_weights.reshape(B, H, W).unsqueeze(1)

        return anomaly_weights


class AttributeScalingResidualGenerator(nn.Module):
    """
    Attribute-scaling Residual Generator (ARG).

    Generates adaptive channel-wise residuals by learning scaling weights
    applied to the input features. Uses tanh to keep residuals in (-1, 1) range,
    producing moderate perturbations that preserve the pre-trained model's integrity.

    Args:
        embed_dim: Feature dimension (C) from the teacher model.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # MLP to generate channel-wise scaling weights
        self.scale_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),  # Output in (-1, 1)
        )

    def forward(self, teacher_features):
        """
        Args:
            teacher_features: (B, C, H, W) — features from frozen teacher block.

        Returns:
            residuals: (B, C, H, W) — adaptive residuals.
        """
        B, C, H, W = teacher_features.shape

        # Reshape to (B, N, C)
        x = teacher_features.flatten(2).transpose(1, 2)  # (B, N, C)

        # Generate scaling weights
        scale_weights = self.scale_net(x)  # (B, N, C) values in (-1, 1)

        # Residual = scale_weights ⊙ input_features
        residuals = scale_weights * x  # (B, N, C)

        # Reshape back to (B, C, H, W)
        residuals = residuals.transpose(1, 2).reshape(B, C, H, W)

        return residuals


class ResidualAnomalyAmplification(nn.Module):
    """
    Complete RAA Module combining MRG and ARG.

    Advanced_Feature = Teacher_Feature + w_a * Residual

    During Stage 1 training:
      - Uses ground truth anomaly mask (w_bar) instead of predicted w_a for
        the residual generator (to avoid interfering with gate training).
      - Both MRG and ARG are trained.

    During Stage 2 / Inference:
      - Uses predicted w_a from MRG.
      - Entire module is frozen.

    Args:
        embed_dim: Feature dimension (C).
        num_memory_items: Number of items per memory bank in MRG.
    """

    def __init__(self, embed_dim, num_memory_items=50):
        super().__init__()
        self.gate = MatchingGuidedResidualGate(embed_dim, num_memory_items)
        self.generator = AttributeScalingResidualGenerator(embed_dim)

    def forward(self, teacher_features, gt_mask=None):
        """
        Args:
            teacher_features: (B, C, H, W) — from frozen teacher block.
            gt_mask: (B, 1, H, W) — ground truth anomaly mask (for Stage 1 training).
                     If None, uses predicted anomaly weights from gate.

        Returns:
            advanced_features: (B, C, H, W) — teacher features + gated residuals.
            anomaly_weights: (B, 1, H, W) — predicted anomaly probabilities.
        """
        # Predict anomaly weights
        anomaly_weights = self.gate(teacher_features)  # (B, 1, H, W)

        # Generate residuals
        residuals = self.generator(teacher_features)  # (B, C, H, W)

        # Gate the residuals
        if gt_mask is not None:
            # During Stage 1 training: use ground truth mask for residual gating
            # This avoids the gate's gradient interfering with the generator
            gating_weights = gt_mask  # (B, 1, H, W)
        else:
            # During inference: use predicted weights
            gating_weights = anomaly_weights

        # Advanced_Feature = Teacher_Feature + w * Residual
        advanced_features = teacher_features + gating_weights * residuals

        return advanced_features, anomaly_weights


# =============================================================================
# Losses for Stage 1
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for the residual gate (binary classification: normal vs anomaly).

    Reduces the contribution of easy examples, focusing training on hard negatives.

    Args:
        alpha: Balancing factor for positive class.
        gamma: Focusing parameter (higher = more focus on hard examples).
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_weights, gt_mask):
        """
        Args:
            pred_weights: (B, 1, H, W) — predicted anomaly probabilities.
            gt_mask: (B, 1, H, W) — ground truth binary mask.
        """
        pred = pred_weights.clamp(1e-6, 1 - 1e-6)

        # Binary cross-entropy components
        bce_pos = -gt_mask * torch.log(pred)
        bce_neg = -(1 - gt_mask) * torch.log(1 - pred)

        # Focal weighting
        pt_pos = pred
        pt_neg = 1 - pred

        focal_pos = self.alpha * (1 - pt_pos) ** self.gamma * bce_pos
        focal_neg = (1 - self.alpha) * (1 - pt_neg) ** self.gamma * bce_neg

        loss = (focal_pos + focal_neg).mean()
        return loss


class AnomalyAmplificationLoss(nn.Module):
    """
    Anomaly Amplification Loss (L_A).

    Pushes abnormal features away from normal features using a dynamic
    cosine-similarity margin based on the vanilla teacher's similarity.

    Only applies to abnormal patches — does NOT pull normal patches
    (to preserve pre-trained model integrity).

    Args:
        alpha: Controls the degree of margin reduction (default: 0.3).
    """

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, advanced_features, teacher_features, gt_mask):
        """
        Args:
            advanced_features: (B, C, H, W) — from the RAA module.
            teacher_features: (B, C, H, W) — from frozen teacher (for dynamic margin).
            gt_mask: (B, 1, H, W) — ground truth anomaly mask.

        Returns:
            loss: Scalar — anomaly amplification loss.
        """
        B, C, H, W = advanced_features.shape

        # Flatten spatial dims: (B, C, N)
        adv_flat = advanced_features.flatten(2)   # (B, C, N)
        tea_flat = teacher_features.flatten(2)     # (B, C, N)
        mask_flat = gt_mask.squeeze(1).flatten(1)  # (B, N)

        # Separate normal and anomaly patches per sample
        total_loss = 0.0
        count = 0

        for b in range(B):
            normal_idx = (mask_flat[b] == 0).nonzero(as_tuple=True)[0]
            anomaly_idx = (mask_flat[b] == 1).nonzero(as_tuple=True)[0]

            if len(anomaly_idx) == 0 or len(normal_idx) == 0:
                continue

            # Get features: (C, N_a) and (C, N_n)
            adv_anomaly = adv_flat[b, :, anomaly_idx]   # (C, N_a)
            adv_normal = adv_flat[b, :, normal_idx]      # (C, N_n)
            tea_anomaly = tea_flat[b, :, anomaly_idx]     # (C, N_a)
            tea_normal = tea_flat[b, :, normal_idx]        # (C, N_n)

            # Normalize for cosine similarity
            adv_anomaly_n = F.normalize(adv_anomaly, dim=0)
            adv_normal_n = F.normalize(adv_normal, dim=0)
            tea_anomaly_n = F.normalize(tea_anomaly, dim=0)
            tea_normal_n = F.normalize(tea_normal, dim=0)

            # Similarity: (N_a, N_n)
            sim_advanced = torch.matmul(adv_anomaly_n.T, adv_normal_n)
            sim_teacher = torch.matmul(tea_anomaly_n.T, tea_normal_n)

            # Dynamic margin: S_ref = alpha * S_teacher
            margin = self.alpha * sim_teacher

            # Loss: max(0, sim_advanced - margin)
            # We want sim_advanced < margin (anomalies should be far from normals)
            loss_matrix = F.relu(sim_advanced - margin)
            total_loss += loss_matrix.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=advanced_features.device, requires_grad=True)

        return total_loss / count
