"""
app.py — Brain MRI Diagnostic System
FastAPI backend serving Stitch AI HTML frontend
EfficientNet-B0 · 6 classes · Grad-CAM · PDF Report
"""

import os, uuid, tempfile, torch, torch.nn as nn, numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from config import (
    CLASS_NAMES, CLASS_DISPLAY, CLASS_DESCRIPTIONS,
    NUM_CLASSES, MODEL_PATH, IMG_SIZE, NORM_MEAN, NORM_STD,
)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}.")
        return None
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    m.to(device).eval()
    return m

model = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def generate_gradcam(mdl, image_tensor, original_image):
    features, gradients = [], []
    def fwd_hook(m, i, o):  features.append(o)
    def bwd_hook(m, gi, go): gradients.append(go[0])
    target_layer = mdl.features[-1]
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)
    out = mdl(image_tensor)
    pred_class = out.argmax()
    mdl.zero_grad()
    out[0, pred_class].backward()
    fh.remove(); bh.remove()
    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    cam = torch.relu(torch.sum(weights * features[0], dim=1)).squeeze()
    cam = (cam / (cam.max() + 1e-8)).cpu().detach().numpy()
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((512, 512))
    cam_np  = np.array(cam_img) / 255.0
    orig = np.array(original_image.resize((512, 512))) / 255.0
    if orig.ndim == 2:
        orig = np.stack([orig] * 3, axis=2)
    heatmap = np.stack([cam_np, np.zeros_like(cam_np), np.zeros_like(cam_np)], axis=2)
    out_img = Image.fromarray(np.uint8(np.clip(orig + heatmap, 0, 1) * 255))
    fid = uuid.uuid4().hex[:8]
    out_path = os.path.join(tempfile.gettempdir(), f"gradcam_{fid}.png")
    out_img.save(out_path)
    return out_path

# ── PDF ───────────────────────────────────────────────────────────────────────
def generate_pdf(patient_name, age, gender, top3, orig_path, gc_path):
    rid = uuid.uuid4().hex[:8]
    fname = os.path.join(tempfile.gettempdir(), f"AI_Report_{rid}.pdf")
    doc = SimpleDocTemplate(fname, pagesize=A4)
    sty = getSampleStyleSheet()
    els = []
    els.append(Paragraph("<b>AI Brain MRI Diagnostic Report</b>", sty["Heading1"]))
    els.append(Spacer(1, 0.3 * inch))
    info = [
        ["Patient Name:", patient_name or "—"],
        ["Age:", age or "—"], ["Gender:", gender],
        ["Report ID:", rid],
        ["Date & Time:", datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
    ]
    t = Table(info, colWidths=[150, 300])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
    ]))
    els.append(t); els.append(Spacer(1, 0.4*inch))
    els.append(Paragraph("<b>Differential Diagnoses (Top 3)</b>", sty["Heading2"]))
    dx = [["Rank", "Condition", "Confidence", "Clinical Note"]]
    for r, (c, cf) in enumerate(top3, 1):
        dx.append([f"#{r}", CLASS_DISPLAY.get(c,c), f"{cf:.1f}%", CLASS_DESCRIPTIONS.get(c,"")])
    dt = Table(dx, colWidths=[35, 120, 70, 220])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0d9488")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    els.append(dt); els.append(Spacer(1, 0.4*inch))
    if os.path.exists(orig_path):
        els.append(Paragraph("<b>Uploaded MRI Scan</b>", sty["Heading3"]))
        els.append(RLImage(orig_path, width=3*inch, height=3*inch))
        els.append(Spacer(1, 0.3*inch))
    if gc_path and os.path.exists(gc_path):
        els.append(Paragraph("<b>Grad-CAM Attention Map</b>", sty["Heading3"]))
        els.append(RLImage(gc_path, width=3*inch, height=3*inch))
        els.append(Spacer(1, 0.3*inch))
    els.append(Paragraph(
        "Disclaimer: AI-generated report for research/educational purposes only. "
        "Not a substitute for professional medical diagnosis.", sty["Normal"]))
    doc.build(els)
    return fname


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Brain MRI Diagnostic System")

# Serve temp files (gradcam images, PDFs, uploaded scans)
app.mount("/static", StaticFiles(directory="static"), name="static")

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the Stitch AI HTML page."""
    with open(TEMPLATE_PATH, "r") as f:
        return f.read()


@app.post("/api/analyse")
async def analyse(
    file: UploadFile = File(...),
    patient_name: str = Form(""),
    age: str = Form(""),
    gender: str = Form("Male"),
    clinical_notes: str = Form(""),
):
    """Run inference, Grad-CAM, PDF — return JSON for the frontend."""
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    # Read uploaded image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Save original
    fid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(tempfile.gettempdir(), f"orig_{fid}.png")
    image.save(orig_path)

    # Inference
    img_t = transform(image).unsqueeze(0).to(device)
    gc_path = generate_gradcam(model, img_t.clone(), image)

    with torch.no_grad():
        probs = torch.softmax(model(img_t), dim=1)[0]
    pnp = probs.cpu().numpy()
    top3_idx = pnp.argsort()[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(pnp[i]*100)) for i in top3_idx]
    top_cls, top_cf = top3[0]
    disp = CLASS_DISPLAY.get(top_cls, top_cls)
    is_normal = top_cls == "Normal"

    study_id = f"#BRN-{uuid.uuid4().hex[:5].upper()}"
    now = datetime.now()

    # PDF
    pdf_path = generate_pdf(patient_name, age, gender, top3, orig_path, gc_path)
    pdf_id = os.path.basename(pdf_path)

    # Store paths for serving
    _temp_files[f"orig_{fid}"] = orig_path
    _temp_files[f"gc_{fid}"] = gc_path
    _temp_files[pdf_id] = pdf_path

    return {
        "patient_name": patient_name,
        "age": age,
        "gender": gender,
        "scan_time": now.strftime("%b %d, %Y") + " • " + now.strftime("%I:%M %p"),
        "study_id": study_id,
        "is_normal": is_normal,
        "diagnosis": disp,
        "confidence": top_cf,
        "original_url": f"/files/orig_{fid}",
        "gradcam_url": f"/files/gc_{fid}",
        "pdf_url": f"/files/{pdf_id}",
        "top3": [
            {
                "name": CLASS_DISPLAY.get(c, c),
                "confidence": cf,
                "note": CLASS_DESCRIPTIONS.get(c, ""),
            }
            for c, cf in top3
        ],
        "all_probs": [
            {
                "name": CLASS_DISPLAY.get(c, c),
                "value": float(pnp[i] * 100),
            }
            for i, c in enumerate(CLASS_NAMES)
        ],
        "insight": CLASS_DESCRIPTIONS.get(top_cls, "No additional insight available."),
    }


# Simple in-memory file store for temp files
_temp_files: dict[str, str] = {}


@app.get("/files/{file_id}")
async def serve_file(file_id: str):
    """Serve a temporary file (image or PDF)."""
    path = _temp_files.get(file_id)
    if path and os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
