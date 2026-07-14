"""
Inference script for the AAND-augmented Teacher-Student framework.

Supports both:
  - Vanilla mode: original DINOv2Teacher + Student (--mode vanilla)
  - AAND mode: AdvancedDINOv2Teacher + Student (--mode aand, default)

Generates anomaly heatmaps, overlays them on original images,
and computes image-level AUROC.
"""

import os
import argparse
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

from models import DINOv2Teacher, AdvancedDINOv2Teacher, CustomStudentCNN


IMAGE_SIZE = 518


class InferenceDataset(Dataset):
    """Simple dataset for inference that loads all images from test/valid splits."""

    def __init__(self, root_dir, splits=None):
        self.samples = []
        self.labels = []

        if splits is None:
            splits = ['test', 'valid']

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for split in splits:
            images_dir = os.path.join(root_dir, split, 'images')
            labels_dir = os.path.join(root_dir, split, 'labels')
            if not os.path.exists(images_dir):
                continue
            for img_path in sorted(glob.glob(os.path.join(images_dir, '*.*'))):
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                base = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, base + '.txt')
                label = 1 if (os.path.exists(label_path) and os.path.getsize(label_path) > 0) else 0
                self.samples.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor, self.labels[idx], img_path


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    if args.mode == 'aand':
        teacher = AdvancedDINOv2Teacher(size='large').to(device)

        raa_path = os.path.join(args.save_dir, 'best_raa.pth')
        if os.path.exists(raa_path):
            teacher.raa_modules.load_state_dict(torch.load(raa_path, map_location=device))
            print(f"Loaded RAA weights from {raa_path}")
        else:
            print("WARNING: No RAA weights found. Using vanilla teacher features.")

        teacher.freeze_raa()
        teacher.eval()

        student_ckpt = os.path.join(args.save_dir, 'best_student_aand.pth')
    else:
        teacher = DINOv2Teacher(size='large').to(device)
        student_ckpt = os.path.join(args.save_dir, 'best_student.pth')

    student = CustomStudentCNN(out_channels=1024).to(device)
    if os.path.exists(student_ckpt):
        student.load_state_dict(torch.load(student_ckpt, map_location=device))
        print(f"Loaded student weights from {student_ckpt}")
    else:
        raise FileNotFoundError(f"Student checkpoint not found at {student_ckpt}")
    student.eval()

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    dataset = InferenceDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)

    image_scores = []
    image_labels = []

    # -------------------------------------------------------------------------
    # Inference loop
    # -------------------------------------------------------------------------
    print(f"\nRunning inference on {len(dataset)} images...")
    pbar = tqdm(dataloader)

    for i, (image, label, img_path) in enumerate(pbar):
        image = image.to(device)
        label = label.item()
        img_path = img_path[0]

        with torch.no_grad():
            # Teacher features
            if args.mode == 'aand':
                teacher_features, _, _ = teacher(image, gt_mask=None)
            else:
                teacher_features = teacher(image)

            teacher_features = F.normalize(teacher_features, dim=1, p=2)

            # Student features
            student_features = student(image)
            student_features = F.normalize(student_features, dim=1, p=2)

            # Anomaly map: per-patch MSE
            diff = (teacher_features - student_features) ** 2
            anomaly_map = torch.mean(diff, dim=1, keepdim=True)  # (1, 1, 37, 37)

            # Upsample to original resolution
            anomaly_map = F.interpolate(
                anomaly_map, size=(IMAGE_SIZE, IMAGE_SIZE),
                mode='bilinear', align_corners=False
            )
            anomaly_map = anomaly_map.squeeze().cpu().numpy()

            # Image-level score
            image_score = np.max(anomaly_map)
            image_scores.append(image_score)
            image_labels.append(label)

            # Visualization
            map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            map_uint8 = (map_norm * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET)

            orig_img = cv2.imread(img_path)
            orig_img = cv2.resize(orig_img, (IMAGE_SIZE, IMAGE_SIZE))

            overlay = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)

            filename = os.path.basename(img_path)
            prefix = "good_" if label == 0 else "anomaly_"
            cv2.imwrite(os.path.join(args.output_dir, f"{prefix}{i}_{filename}"), overlay)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    if len(set(image_labels)) > 1:
        auroc = roc_auc_score(image_labels, image_scores)
        print(f"\n{'='*40}")
        print(f"Image-level AUROC: {auroc:.4f}")
        print(f"{'='*40}")
    else:
        print("\nOnly one class present in test set, skipping AUROC calculation.")

    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for anomaly detection")
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Path to the dataset root")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory where model weights are saved")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save anomaly maps")
    parser.add_argument("--mode", type=str, default="aand",
                        choices=["vanilla", "aand"],
                        help="Inference mode: vanilla (original) or aand (enhanced)")
    args = parser.parse_args()

    infer(args)
