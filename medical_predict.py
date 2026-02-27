"""
medical_predict.py — CLI prediction script for the 6-class Brain MRI model.

Usage:
    python3 medical_predict.py                        # uses uploaded_image.png
    python3 medical_predict.py --image path/to/scan.png
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from pdf_report import generate_medical_report
from config import (
    CLASS_NAMES, CLASS_DISPLAY, CLASS_DESCRIPTIONS,
    NUM_CLASSES, MODEL_PATH, IMG_SIZE, NORM_MEAN, NORM_STD
)

HEATMAP_PATH = "heatmap.png"
OUTPUT_PDF   = "AI_Medical_Report.pdf"

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Model ──────────────────────────────────────────────────────────────────────
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ── Transform ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def generate_gradcam_heatmap(image_tensor, original_image_path, save_path):
    import cv2, numpy as np
    gradients, activations = [], []

    def bwd_hook(m, gi, go): gradients.append(go[0])
    def fwd_hook(m, i, o):   activations.append(o)

    target_layer   = model.features[-1]
    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    output = model(image_tensor)
    pred   = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred].backward()

    fwd_h.remove()
    bwd_h.remove()

    grads  = gradients[0]
    acts   = activations[0]
    w      = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= w[i]

    heatmap = torch.relu(torch.mean(acts, dim=1)).squeeze()
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap / (heatmap.max() + 1e-8))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(original_image_path)
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, superimposed)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="uploaded_image.png", help="Path to MRI image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    image  = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    # Grad-CAM (before no_grad)
    generate_gradcam_heatmap(tensor.clone(), args.image, HEATMAP_PATH)

    # Prediction
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top3_idx = probs.argsort()[::-1][:3]
    top3     = [(CLASS_NAMES[i], float(probs[i] * 100)) for i in top3_idx]

    top_class, top_conf = top3[0]

    print("\n🧠 Brain MRI Diagnosis")
    print("─" * 45)
    for rank, (cls, conf) in enumerate(top3, 1):
        display = CLASS_DISPLAY.get(cls, cls)
        print(f"  #{rank}  {display:<30}  {conf:5.1f}%")
    print("─" * 45)

    generate_medical_report(
        image_path          = args.image,
        prediction          = top_class,
        confidence          = top_conf,
        heatmap_path        = HEATMAP_PATH,
        output_pdf          = OUTPUT_PDF,
        top3                = top3,
        class_display       = CLASS_DISPLAY,
        class_descriptions  = CLASS_DESCRIPTIONS,
    )
    print(f"\n📄 Report saved: {OUTPUT_PDF}")
