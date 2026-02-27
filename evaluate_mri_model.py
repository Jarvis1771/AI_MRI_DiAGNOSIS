"""
evaluate_mri_model.py — Evaluate the 6-class Brain MRI model on the validation set.
Prints per-class precision, recall, F1 and full confusion matrix.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from config import CLASS_NAMES, NUM_CLASSES, MODEL_PATH, VAL_DIR, IMG_SIZE, NORM_MEAN, NORM_STD

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Transform ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# ── Dataset ────────────────────────────────────────────────────────────────────
dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
loader  = DataLoader(dataset, batch_size=16, shuffle=False)

print(f"Classes (folder): {dataset.classes}")
print(f"Classes (config): {CLASS_NAMES}")

if dataset.classes != CLASS_NAMES:
    print("\n⚠️  Class order mismatch between val folder and config.py!")
    print("   Results may be incorrect. Check folder names.\n")

# ── Model ──────────────────────────────────────────────────────────────────────
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ── Evaluation ─────────────────────────────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds   = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ── Results ────────────────────────────────────────────────────────────────────
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
# Pretty-print with class labels
header = f"{'':>20}" + "".join(f"{c:>20}" for c in CLASS_NAMES)
print(header)
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:>20}" + "".join(f"{v:>20}" for v in row))

print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))

overall_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) * 100
print(f"✅ Overall Accuracy: {overall_acc:.2f}%")
