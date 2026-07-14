"""
Two-stage training script for the AAND-augmented Teacher-Student framework.

Stage 1: Anomaly Amplification
  - Train RAA modules with synthetic anomalies (Focal Loss + Amplification Loss).
  - DINOv2 backbone is frozen; only RAA parameters are updated.

Stage 2: Normality Distillation
  - Train Student CNN against the Advanced Teacher with HKD loss.
  - Advanced Teacher (backbone + RAA) is fully frozen.
  - Student learns to reconstruct only normal features.

Reference: AAND paper (arxiv 2405.02068v2)
"""

import os
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from models import AdvancedDINOv2Teacher, CustomStudentCNN
from anomaly_synthesis import AnomalySynthesizer
from raa_module import FocalLoss, AnomalyAmplificationLoss


# =============================================================================
# Dataset for Stage 1 (with anomaly synthesis)
# =============================================================================

class Stage1Dataset(Dataset):
    """
    Dataset for Stage 1 training. For each normal image, generates a
    synthetic anomaly on-the-fly and returns both the corrupted image
    and the anomaly mask.
    """

    def __init__(self, root_dir, image_size=518, texture_source='procedural',
                 dtd_path=None, splits=None):
        self.image_size = image_size
        self.image_paths = []

        if splits is None:
            splits = ['train']

        for split in splits:
            images_dir = os.path.join(root_dir, split, 'images')
            labels_dir = os.path.join(root_dir, split, 'labels')
            if not os.path.exists(images_dir):
                continue

            for img_path in glob.glob(os.path.join(images_dir, '*.*')):
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                # For Stage 1 we can use ALL images (even ones with defects)
                # because we generate our own synthetic anomalies
                self.image_paths.append(img_path)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.synthesizer = AnomalySynthesizer(
            image_size=image_size,
            texture_source=texture_source,
            dtd_path=dtd_path
        )

        print(f"Stage1Dataset: {len(self.image_paths)} images loaded")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Generate synthetic anomaly
        corrupted_image, anomaly_mask = self.synthesizer(image)

        # Apply transforms
        corrupted_tensor = self.transform(corrupted_image)

        # Downsample mask to feature resolution (37x37 for 518px input with patch14)
        feat_h = self.image_size // 14
        feat_w = self.image_size // 14
        mask_tensor = torch.from_numpy(anomaly_mask).float().unsqueeze(0)  # (1, H, W)
        mask_downsampled = F.interpolate(
            mask_tensor.unsqueeze(0), size=(feat_h, feat_w), mode='nearest'
        ).squeeze(0)  # (1, feat_h, feat_w)

        return corrupted_tensor, mask_downsampled


class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 training. Only returns normal images.
    """

    def __init__(self, root_dir, image_size=518, splits=None):
        self.image_size = image_size
        self.image_paths = []

        if splits is None:
            splits = ['train']

        for split in splits:
            images_dir = os.path.join(root_dir, split, 'images')
            labels_dir = os.path.join(root_dir, split, 'labels')
            if not os.path.exists(images_dir):
                continue

            for img_path in glob.glob(os.path.join(images_dir, '*.*')):
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                # Check if this is a normal image (empty or missing label file)
                base = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, base + '.txt')
                if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                    self.image_paths.append(img_path)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Stage2Dataset: {len(self.image_paths)} normal images loaded")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor


# =============================================================================
# Hard Knowledge Distillation Loss
# =============================================================================

class HardKnowledgeDistillationLoss(nn.Module):
    """
    Hard Knowledge Distillation (HKD) Loss.

    Standard KD loss (cosine similarity) + extra emphasis on the top-K_h
    hardest normal patches (highest reconstruction error).

    This forces the student to pay more attention to fine-grained textures
    and rare normal patterns that are hard to reconstruct.

    Args:
        k_hard: Number of hard samples to select (K_h in the paper).
        lambda_hkd: Weight for the hard sample loss component.
    """

    def __init__(self, k_hard=10, lambda_hkd=1.0):
        super().__init__()
        self.k_hard = k_hard
        self.lambda_hkd = lambda_hkd

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, C, H, W) — from student model.
            teacher_features: (B, C, H, W) — from advanced teacher.

        Returns:
            total_loss: Standard KD loss + weighted HKD loss.
            kd_loss: Standard cosine distillation loss (for logging).
            hkd_loss: Hard knowledge distillation loss (for logging).
        """
        B, C, H, W = student_features.shape

        # Normalize features
        s_norm = F.normalize(student_features, dim=1, p=2)
        t_norm = F.normalize(teacher_features, dim=1, p=2)

        # Standard KD loss: 1 - cosine_similarity, averaged over all patches
        # Cosine similarity per patch: (B, H, W)
        cos_sim = (s_norm * t_norm).sum(dim=1)  # (B, H, W)
        kd_loss = (1 - cos_sim).mean()

        # HKD: select top-K_h patches with highest reconstruction error per sample
        patch_errors = (1 - cos_sim).flatten(1)  # (B, N) where N = H*W
        k = min(self.k_hard, patch_errors.shape[1])

        # Top-K hardest patches
        topk_errors, _ = torch.topk(patch_errors, k, dim=1)  # (B, K_h)
        hkd_loss = topk_errors.mean()

        total_loss = kd_loss + self.lambda_hkd * hkd_loss

        return total_loss, kd_loss, hkd_loss


# =============================================================================
# Training Functions
# =============================================================================

def train_stage1(args):
    """Stage 1: Anomaly Amplification — train RAA modules."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"STAGE 1: ANOMALY AMPLIFICATION")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    # Model
    teacher = AdvancedDINOv2Teacher(
        size='large',
        num_memory_items=args.num_memory_items
    ).to(device)
    teacher.unfreeze_raa()

    # Dataset
    dataset = Stage1Dataset(
        root_dir=args.data_path,
        image_size=518,
        texture_source=args.texture_source,
        dtd_path=args.dtd_path,
        splits=['train']
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Only optimize RAA parameters
    raa_params = list(teacher.raa_modules.parameters())
    optimizer = optim.Adam(raa_params, lr=args.lr_stage1)

    # Losses
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    amp_loss_fn = AnomalyAmplificationLoss(alpha=args.margin_alpha)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    print(f"\nTraining RAA modules for {args.epochs_stage1} epochs...")
    print(f"  RAA trainable params: {sum(p.numel() for p in raa_params):,}")

    for epoch in range(args.epochs_stage1):
        teacher.raa_modules.train()
        epoch_focal = 0.0
        epoch_amp = 0.0
        epoch_total = 0.0

        pbar = tqdm(dataloader, desc=f"S1 Epoch {epoch+1}/{args.epochs_stage1}")
        for corrupted_images, gt_masks in pbar:
            corrupted_images = corrupted_images.to(device)
            gt_masks = gt_masks.to(device)  # (B, 1, 37, 37)

            # Forward through advanced teacher
            adv_features, pred_weights, vanilla_features = teacher(
                corrupted_images, gt_mask=gt_masks
            )

            # Focal loss on predicted anomaly weights
            l_focal = focal_loss_fn(pred_weights, gt_masks)

            # Anomaly amplification loss
            l_amp = amp_loss_fn(adv_features, vanilla_features, gt_masks)

            loss = l_focal + l_amp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_focal += l_focal.item()
            epoch_amp += l_amp.item()
            epoch_total += loss.item()
            pbar.set_postfix({
                'focal': f'{l_focal.item():.4f}',
                'amp': f'{l_amp.item():.4f}',
                'total': f'{loss.item():.4f}'
            })

        n = len(dataloader)
        avg_total = epoch_total / n
        print(f"Epoch {epoch+1} | Focal: {epoch_focal/n:.4f} | Amp: {epoch_amp/n:.4f} | Total: {avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(teacher.raa_modules.state_dict(),
                       os.path.join(args.save_dir, 'best_raa.pth'))
            print("  → Saved best RAA checkpoint!")

    # Save final
    torch.save(teacher.raa_modules.state_dict(),
               os.path.join(args.save_dir, 'final_raa.pth'))
    print(f"\nStage 1 complete. Best loss: {best_loss:.4f}")

    return teacher


def train_stage2(args, teacher=None):
    """Stage 2: Normality Distillation — train Student with HKD loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"STAGE 2: NORMALITY DISTILLATION")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    # Load Advanced Teacher if not provided
    if teacher is None:
        teacher = AdvancedDINOv2Teacher(
            size='large',
            num_memory_items=args.num_memory_items
        ).to(device)

        raa_path = os.path.join(args.save_dir, 'best_raa.pth')
        if os.path.exists(raa_path):
            teacher.raa_modules.load_state_dict(torch.load(raa_path, map_location=device))
            print(f"Loaded RAA weights from {raa_path}")
        else:
            print("WARNING: No RAA weights found. Using randomly initialized RAA.")

    # Freeze entire teacher (including RAA)
    teacher.freeze_raa()
    teacher.eval()

    # Student model
    student = CustomStudentCNN(out_channels=1024).to(device)

    # Dataset (normal images only)
    dataset = Stage2Dataset(
        root_dir=args.data_path,
        image_size=518,
        splits=['train']
    )

    if len(dataset) == 0:
        print("WARNING: No normal images found in train split. Trying all splits...")
        dataset = Stage2Dataset(
            root_dir=args.data_path,
            image_size=518,
            splits=['train', 'valid', 'test']
        )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=args.lr_stage2, weight_decay=1e-4)

    # HKD Loss
    hkd_loss_fn = HardKnowledgeDistillationLoss(
        k_hard=args.k_hard,
        lambda_hkd=args.lambda_hkd
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    print(f"\nTraining Student for {args.epochs_stage2} epochs...")
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")
    print(f"  K_hard: {args.k_hard}, λ_hkd: {args.lambda_hkd}")

    for epoch in range(args.epochs_stage2):
        student.train()
        epoch_kd = 0.0
        epoch_hkd = 0.0
        epoch_total = 0.0

        pbar = tqdm(dataloader, desc=f"S2 Epoch {epoch+1}/{args.epochs_stage2}")
        for images in pbar:
            images = images.to(device)

            # Teacher features (frozen)
            with torch.no_grad():
                adv_features, _, _ = teacher(images, gt_mask=None)
                # Detach to ensure no gradients flow to teacher
                teacher_features = adv_features.detach()

            # Student features
            student_features = student(images)

            # HKD loss
            total_loss, kd_loss, hkd_loss = hkd_loss_fn(student_features, teacher_features)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_kd += kd_loss.item()
            epoch_hkd += hkd_loss.item()
            epoch_total += total_loss.item()
            pbar.set_postfix({
                'kd': f'{kd_loss.item():.4f}',
                'hkd': f'{hkd_loss.item():.4f}',
                'total': f'{total_loss.item():.4f}'
            })

        n = len(dataloader)
        avg_total = epoch_total / n
        print(f"Epoch {epoch+1} | KD: {epoch_kd/n:.4f} | HKD: {epoch_hkd/n:.4f} | Total: {avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(student.state_dict(),
                       os.path.join(args.save_dir, 'best_student_aand.pth'))
            print("  → Saved best Student checkpoint!")

    # Save final
    torch.save(student.state_dict(),
               os.path.join(args.save_dir, 'final_student_aand.pth'))
    print(f"\nStage 2 complete. Best loss: {best_loss:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AAND Two-Stage Training")

    # General
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Path to the dataset root")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (smaller due to DINOv2-L memory)")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save weights")

    # Stage 1: Anomaly Amplification
    parser.add_argument("--epochs_stage1", type=int, default=100,
                        help="Number of epochs for Stage 1")
    parser.add_argument("--lr_stage1", type=float, default=5e-3,
                        help="Learning rate for Stage 1 (RAA modules)")
    parser.add_argument("--num_memory_items", type=int, default=50,
                        help="Number of items per memory bank in MRG (L)")
    parser.add_argument("--margin_alpha", type=float, default=0.3,
                        help="Dynamic margin factor for amplification loss (α)")
    parser.add_argument("--texture_source", type=str, default="procedural",
                        choices=["procedural", "dtd"],
                        help="Texture source for anomaly synthesis")
    parser.add_argument("--dtd_path", type=str, default=None,
                        help="Path to DTD dataset (required if texture_source='dtd')")

    # Stage 2: Normality Distillation
    parser.add_argument("--epochs_stage2", type=int, default=120,
                        help="Number of epochs for Stage 2")
    parser.add_argument("--lr_stage2", type=float, default=1e-3,
                        help="Learning rate for Stage 2 (Student)")
    parser.add_argument("--k_hard", type=int, default=10,
                        help="Number of hard samples for HKD loss (K_h)")
    parser.add_argument("--lambda_hkd", type=float, default=1.0,
                        help="Weight for hard knowledge distillation loss")

    # Control
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage 1 (use existing RAA weights)")
    parser.add_argument("--skip_stage2", action="store_true",
                        help="Skip Stage 2 (only run Stage 1)")

    args = parser.parse_args()

    teacher = None
    if not args.skip_stage1:
        teacher = train_stage1(args)

    if not args.skip_stage2:
        train_stage2(args, teacher=teacher)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Checkpoints saved to: {args.save_dir}/")
    print(f"  - best_raa.pth        (Advanced Teacher RAA weights)")
    print(f"  - best_student_aand.pth (Student weights)")


if __name__ == "__main__":
    main()
