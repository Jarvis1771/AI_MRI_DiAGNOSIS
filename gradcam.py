"""
gradcam.py — Standalone Grad-CAM visualiser for the 6-class Brain MRI model.
Shows top-3 predictions + saves heatmap overlay.

Usage:
    python3 gradcam.py
    python3 gradcam.py --image path/to/scan.jpg
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

from config import (
    CLASS_NAMES, CLASS_DISPLAY, NUM_CLASSES,
    MODEL_PATH, IMG_SIZE, NORM_MEAN, NORM_STD
)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_IMAGE  = "uploaded_image.png"
OUTPUT_PATH    = "heatmap.png"
DEVICE         = torch.device("cpu")   # Grad-CAM standalone; CPU is fine

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    m.to(DEVICE)
    return m

# ── Grad-CAM class ────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam     = torch.relu(torch.sum(weights * self.activations, dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

# ── Image loading ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

def load_image(path):
    img = Image.open(path).convert("L")
    return img, transform(img).unsqueeze(0).to(DEVICE)

# ── Main ──────────────────────────────────────────────────────────────────────
def generate_gradcam(image_path=DEFAULT_IMAGE):
    model   = load_model()
    img, tensor = load_image(image_path)

    # Top-3 predictions
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top3_idx = probs.argsort()[::-1][:3]

    print("\n🧠 Brain MRI Analysis")
    print("─" * 40)
    for rank, idx in enumerate(top3_idx, 1):
        name = CLASS_DISPLAY.get(CLASS_NAMES[idx], CLASS_NAMES[idx])
        print(f"  #{rank}  {name:<30} {probs[idx]*100:5.1f}%")
    print("─" * 40)

    # Grad-CAM for top-1
    gradcam = GradCAM(model, model.features[-1])
    cam     = gradcam.generate(tensor, top3_idx[0])

    img_np  = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"\n🔥 Grad-CAM heatmap saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to MRI image")
    args = parser.parse_args()
    generate_gradcam(args.image)
