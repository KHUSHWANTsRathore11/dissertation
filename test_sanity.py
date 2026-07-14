"""Quick sanity check for the AAND pipeline."""
import torch
import torch.nn.functional as F
from models import AdvancedDINOv2Teacher, CustomStudentCNN
from anomaly_synthesis import AnomalySynthesizer, generate_perlin_mask
from raa_module import FocalLoss, AnomalyAmplificationLoss
from train_aand import HardKnowledgeDistillationLoss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Test 1: Perlin noise mask ---
print("\n[1/5] Perlin noise mask generation...")
mask = generate_perlin_mask(518, 518)
print(f"  Mask shape: {mask.shape}, unique values: {np.unique(mask)}, ratio: {mask.mean():.3f}")
assert mask.shape == (518, 518), "Mask shape mismatch"
print("  ✅ PASS")

# --- Test 2: Anomaly Synthesizer ---
print("\n[2/5] Anomaly Synthesizer (procedural)...")
from PIL import Image
dummy_img = Image.fromarray(np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8))
synth = AnomalySynthesizer(image_size=518, texture_source='procedural')
corrupted, amask = synth(dummy_img)
print(f"  Corrupted: {corrupted.size}, Mask shape: {amask.shape}")
assert amask.shape == (518, 518), "Anomaly mask shape mismatch"
print("  ✅ PASS")

# --- Test 3: Advanced Teacher forward pass ---
print("\n[3/5] AdvancedDINOv2Teacher forward pass...")
teacher = AdvancedDINOv2Teacher(size='large').to(device)
dummy_input = torch.randn(2, 3, 518, 518).to(device)
gt_mask = torch.randint(0, 2, (2, 1, 37, 37)).float().to(device)

adv_feat, pred_w, vanilla_feat = teacher(dummy_input, gt_mask=gt_mask)
print(f"  Advanced features: {adv_feat.shape}")
print(f"  Predicted weights: {pred_w.shape}")
print(f"  Vanilla features:  {vanilla_feat.shape}")
assert adv_feat.shape == (2, 1024, 37, 37), f"Advanced feature shape mismatch: {adv_feat.shape}"
assert pred_w.shape == (2, 1, 37, 37), f"Weight shape mismatch: {pred_w.shape}"
print("  ✅ PASS")

# --- Test 4: Student forward pass + shape matching ---
print("\n[4/5] Student forward pass + shape compatibility...")
student = CustomStudentCNN(out_channels=1024).to(device)
student_feat = student(dummy_input)
print(f"  Student features: {student_feat.shape}")
assert student_feat.shape == adv_feat.shape, "Student/Teacher shape mismatch!"
print("  ✅ PASS")

# --- Test 5: Loss computations ---
print("\n[5/5] Loss computations...")
focal = FocalLoss()
l_focal = focal(pred_w, gt_mask)
print(f"  Focal loss: {l_focal.item():.4f}")

amp = AnomalyAmplificationLoss()
l_amp = amp(adv_feat, vanilla_feat, gt_mask)
print(f"  Amplification loss: {l_amp.item():.4f}")

hkd = HardKnowledgeDistillationLoss(k_hard=10)
s_norm = F.normalize(student_feat, dim=1, p=2)
t_norm = F.normalize(adv_feat, dim=1, p=2)
l_total, l_kd, l_hkd = hkd(s_norm, t_norm)
print(f"  KD loss: {l_kd.item():.4f}, HKD loss: {l_hkd.item():.4f}, Total: {l_total.item():.4f}")

# Check gradients flow correctly
l_total.backward()
grad_check = any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
print(f"  Gradients flow to student: {grad_check}")
assert grad_check, "No gradients flowing to student!"
print("  ✅ PASS")

print(f"\n{'='*50}")
print("ALL SANITY CHECKS PASSED ✅")
print(f"{'='*50}")
