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
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

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

# ── Plot training curves ──────────────────────────────────────────────────────
CURVES_PATH = "training_curves.png"

def plot_training_curves(history):
    """Save a dual-panel figure: Loss vs Epochs + Accuracy vs Epochs."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curve ────────────────────────────────────────────────────────
    ax1.plot(epochs, history["loss"], 'o-', color='#dc2626', linewidth=2,
             markersize=5, label='Training Loss')
    ax1.fill_between(epochs, history["loss"], alpha=0.1, color='#dc2626')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss vs Epochs', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))

    # ── Accuracy curve ────────────────────────────────────────────────────
    ax2.plot(epochs, history["val_acc"], 'o-', color='#0d9488', linewidth=2,
             markersize=5, label='Validation Accuracy')
    ax2.fill_between(epochs, history["val_acc"], alpha=0.1, color='#0d9488')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy vs Epochs', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(min(history["val_acc"]) - 2, 101)

    # Mark best epoch
    best_idx = np.argmax(history["val_acc"])
    ax2.annotate(f'Best: {history["val_acc"][best_idx]:.2f}%',
                 xy=(best_idx + 1, history["val_acc"][best_idx]),
                 xytext=(best_idx + 1, history["val_acc"][best_idx] - 3),
                 fontsize=9, fontweight='bold', color='#0d9488',
                 arrowprops=dict(arrowstyle='->', color='#0d9488'),
                 ha='center')

    fig.suptitle('EfficientNet-B0 · Brain MRI 6-Class Classifier',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"📈 Training curves saved to: {CURVES_PATH}")
    plt.close()


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
    history = {"loss": [], "val_acc": []}
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
        history["loss"].append(avg_loss)
        history["val_acc"].append(acc)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%", end="")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Saved (best so far)")
        else:
            print()

    # Save training curves plot
    plot_training_curves(history)

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
