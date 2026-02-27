"""
train_brain_mri.py — Train a 6-class Brain MRI classifier (EfficientNet-B0).

Expected data layout:
    data/mri/train/{Alzheimers, BrainTumor, Hemorrhage, MultipleSclerosis, Normal, Stroke}/
    data/mri/val/  {same 6 folders}

Usage:
    python3 train_brain_mri.py            # full training
    python3 train_brain_mri.py --check-only   # dry-run: validates data + model, no training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np

from config import (
    CLASS_NAMES, NUM_CLASSES, MODEL_PATH,
    TRAIN_DIR, VAL_DIR, IMG_SIZE, NORM_MEAN, NORM_STD
)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS     = 20
LR         = 1e-4

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

os.makedirs("models", exist_ok=True)

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # MRI → 3-channel for EfficientNet
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# ── Dataset validation ────────────────────────────────────────────────────────
def validate_dataset_structure():
    """Check all 6 class folders exist in train and val."""
    missing = []
    for split in [TRAIN_DIR, VAL_DIR]:
        for cls in CLASS_NAMES:
            folder = os.path.join(split, cls)
            if not os.path.isdir(folder):
                missing.append(folder)

    if missing:
        print("\n❌ Missing dataset folders:")
        for m in missing:
            print(f"   {m}")
        print("\n📌 Create these folders, add MRI images, and run again.")
        print("   Recommended datasets: see implementation_plan.md")
        sys.exit(1)

# ── WeightedRandomSampler ─────────────────────────────────────────────────────
def make_weighted_sampler(dataset):
    """Returns a sampler that up-samples minority classes."""
    class_counts = np.array([0] * NUM_CLASSES)
    for _, label in dataset:
        class_counts[label] += 1

    class_counts = np.maximum(class_counts, 1)  # avoid div-by-zero
    weights_per_class = 1.0 / class_counts
    sample_weights = [weights_per_class[label] for _, label in dataset]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(device)

# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    validate_dataset_structure()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

    # Verify class order matches config.py
    if train_dataset.classes != CLASS_NAMES:
        print(f"\n⚠️  Class order mismatch!")
        print(f"   config.py : {CLASS_NAMES}")
        print(f"   folder    : {train_dataset.classes}")
        print("   Rename folders to match config.py exactly and retry.")
        sys.exit(1)

    print(f"\n✅ Dataset loaded")
    print(f"   Classes      : {train_dataset.classes}")
    print(f"   Train images : {len(train_dataset)}")
    print(f"   Val images   : {len(val_dataset)}")

    sampler     = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    print(f"\n🚀 Training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        scheduler.step()

        acc      = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%", end="")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Saved (best so far)")
        else:
            print()

    print(f"\n✅ Training complete")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"💾 Model saved to: {MODEL_PATH}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--check-only" in sys.argv:
        print("🔍 DRY RUN — checking dataset structure and model build only\n")
        validate_dataset_structure()
        model = build_model()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        out   = model(dummy)
        print(f"✅ Model output shape: {out.shape}  (expected: [1, {NUM_CLASSES}])")
        print("✅ All checks passed — ready to train!")
    else:
        train()
