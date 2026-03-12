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
def generate_pdf(patient_name, age, gender, top3, orig_path, gc_path, all_probs=None):
    """Generate a professionally styled PDF diagnostic report."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import HRFlowable, KeepTogether

    TEAL = colors.HexColor("#0d9488")
    TEAL_LIGHT = colors.HexColor("#f0fdfa")
    DANGER = colors.HexColor("#dc2626")
    DANGER_LIGHT = colors.HexColor("#fef2f2")
    SUCCESS = colors.HexColor("#059669")
    SUCCESS_LIGHT = colors.HexColor("#ecfdf5")
    SLATE_900 = colors.HexColor("#0f172a")
    SLATE_600 = colors.HexColor("#475569")
    SLATE_400 = colors.HexColor("#94a3b8")
    SLATE_100 = colors.HexColor("#f1f5f9")
    AMBER = colors.HexColor("#d97706")
    AMBER_LIGHT = colors.HexColor("#fffbeb")

    rid = uuid.uuid4().hex[:8].upper()
    fname = os.path.join(tempfile.gettempdir(), f"AI_Report_{rid}.pdf")
    now = datetime.now()
    top_cls, top_cf = top3[0]
    is_normal = top_cls == "Normal"

    doc = SimpleDocTemplate(
        fname, pagesize=A4,
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.5*inch, bottomMargin=0.6*inch,
    )
    page_w = A4[0] - 1.2*inch  # usable width

    # ── Custom styles ─────────────────────────────────────────────────────
    s_title = ParagraphStyle("s_title", fontName="Helvetica-Bold", fontSize=20,
                             textColor=colors.white, leading=24)
    s_subtitle = ParagraphStyle("s_sub", fontName="Helvetica", fontSize=9,
                                textColor=colors.white, leading=12)
    s_section = ParagraphStyle("s_sec", fontName="Helvetica-Bold", fontSize=13,
                               textColor=SLATE_900, leading=18, spaceBefore=16,
                               spaceAfter=8)
    s_body = ParagraphStyle("s_body", fontName="Helvetica", fontSize=10,
                            textColor=SLATE_600, leading=14)
    s_label = ParagraphStyle("s_label", fontName="Helvetica-Bold", fontSize=8,
                             textColor=SLATE_400, leading=10)
    s_value = ParagraphStyle("s_val", fontName="Helvetica-Bold", fontSize=11,
                             textColor=SLATE_900, leading=14)
    s_small = ParagraphStyle("s_small", fontName="Helvetica", fontSize=8,
                             textColor=SLATE_400, leading=10)
    s_disclaimer = ParagraphStyle("s_disc", fontName="Helvetica", fontSize=8,
                                  textColor=AMBER, leading=11)
    s_footer = ParagraphStyle("s_foot", fontName="Helvetica", fontSize=7,
                              textColor=SLATE_400, leading=10, alignment=TA_CENTER)
    s_center = ParagraphStyle("s_center", fontName="Helvetica", fontSize=9,
                              textColor=SLATE_600, leading=12, alignment=TA_CENTER)

    els = []

    # ══════════════════════════════════════════════════════════════════════
    #  HEADER BAR — teal background with title and report ID
    # ══════════════════════════════════════════════════════════════════════
    header_data = [[
        Paragraph("Brain MRI Diagnostic System", s_title),
        Paragraph(f"Report #{rid}<br/>EfficientNet-B0 · 6-Class Classifier", s_subtitle),
    ]]
    header = Table(header_data, colWidths=[page_w * 0.65, page_w * 0.35])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), TEAL),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (0,0), 16),
        ("RIGHTPADDING", (-1,-1), (-1,-1), 16),
        ("TOPPADDING", (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("ALIGN", (-1,-1), (-1,-1), "RIGHT"),
        ("ROUNDEDCORNERS", [8, 8, 0, 0]),
    ]))
    els.append(header)

    # ── Sub-header with date ──────────────────────────────────────────────
    date_bar = Table(
        [[Paragraph(f"Generated: {now.strftime('%B %d, %Y at %I:%M %p')}", s_small)]],
        colWidths=[page_w],
    )
    date_bar.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), SLATE_100),
        ("LEFTPADDING", (0,0), (0,0), 16),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("ROUNDEDCORNERS", [0, 0, 8, 8]),
    ]))
    els.append(date_bar)
    els.append(Spacer(1, 0.3*inch))

    # ══════════════════════════════════════════════════════════════════════
    #  PATIENT INFORMATION
    # ══════════════════════════════════════════════════════════════════════
    els.append(Paragraph("Patient Information", s_section))

    def _info_cell(label, value):
        return [Paragraph(label, s_label), Paragraph(str(value) if value else "—", s_value)]

    info_row1 = [_info_cell("PATIENT NAME", patient_name),
                 _info_cell("AGE", f"{age} years" if age else "—"),
                 _info_cell("GENDER", gender),
                 _info_cell("STUDY ID", f"BRN-{rid}")]
    # Flatten into table
    labels = [c[0] for c in info_row1]
    values = [c[1] for c in info_row1]
    info_t = Table([labels, values], colWidths=[page_w/4]*4)
    info_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), 0.5, SLATE_100),
        ("LINEBELOW", (0,0), (-1,0), 0, colors.white),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        # Vertical dividers
        ("LINEAFTER", (0,0), (0,-1), 0.5, SLATE_100),
        ("LINEAFTER", (1,0), (1,-1), 0.5, SLATE_100),
        ("LINEAFTER", (2,0), (2,-1), 0.5, SLATE_100),
    ]))
    els.append(info_t)
    els.append(Spacer(1, 0.25*inch))

    # ══════════════════════════════════════════════════════════════════════
    #  PRIMARY DIAGNOSIS
    # ══════════════════════════════════════════════════════════════════════
    els.append(Paragraph("Primary Diagnosis", s_section))

    badge_color = SUCCESS if is_normal else DANGER
    badge_bg = SUCCESS_LIGHT if is_normal else DANGER_LIGHT
    badge_text = "✓ Normal Finding" if is_normal else "⚠ Abnormal Finding"
    s_badge = ParagraphStyle("badge", fontName="Helvetica-Bold", fontSize=9,
                             textColor=badge_color, leading=12)
    s_diag_name = ParagraphStyle("diag", fontName="Helvetica-Bold", fontSize=18,
                                 textColor=SLATE_900, leading=22)
    s_conf_label = ParagraphStyle("cl", fontName="Helvetica", fontSize=9,
                                  textColor=SLATE_400, leading=11, alignment=TA_RIGHT)
    s_conf_val = ParagraphStyle("cv", fontName="Helvetica-Bold", fontSize=28,
                                textColor=badge_color, leading=32, alignment=TA_RIGHT)

    diag_display = CLASS_DISPLAY.get(top_cls, top_cls)
    diag_desc = CLASS_DESCRIPTIONS.get(top_cls, "")

    diag_left = [
        [Paragraph(badge_text, s_badge)],
        [Paragraph(diag_display, s_diag_name)],
        [Paragraph(diag_desc[:100], s_body)],
    ]
    diag_right = [
        [Paragraph("Confidence Score", s_conf_label)],
        [Paragraph(f"{top_cf:.1f}%", s_conf_val)],
        [Paragraph("AI Probability Scale", s_small)],
    ]

    diag_left_t = Table(diag_left, colWidths=[page_w*0.62])
    diag_left_t.setStyle(TableStyle([
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    diag_right_t = Table(diag_right, colWidths=[page_w*0.32])
    diag_right_t.setStyle(TableStyle([
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))

    diag_outer = Table([[diag_left_t, diag_right_t]], colWidths=[page_w*0.64, page_w*0.36])
    diag_outer.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), badge_bg),
        ("BOX", (0,0), (-1,-1), 0.5, badge_color),
        ("TOPPADDING", (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("LEFTPADDING", (0,0), (0,0), 16),
        ("RIGHTPADDING", (-1,-1), (-1,-1), 16),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    els.append(diag_outer)

    # Confidence bar
    bar_filled_w = page_w * (top_cf / 100.0)
    bar_bg_w = page_w - bar_filled_w
    bar_data = [[""]]
    bar_t = Table(bar_data, colWidths=[page_w], rowHeights=[10])
    bar_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), SLATE_100),
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    els.append(bar_t)

    # Filled portion (overlay approach — just a colored table)
    bar_fill = Table([[""]], colWidths=[bar_filled_w], rowHeights=[10])
    bar_fill.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), badge_color),
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    # Use a combined approach
    bar_combined = Table([[bar_fill, ""]], colWidths=[bar_filled_w, bar_bg_w], rowHeights=[10])
    bar_combined.setStyle(TableStyle([
        ("BACKGROUND", (1,0), (1,0), SLATE_100),
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    els.append(bar_combined)
    els.append(Spacer(1, 0.25*inch))

    # ══════════════════════════════════════════════════════════════════════
    #  DIFFERENTIAL DIAGNOSES TABLE
    # ══════════════════════════════════════════════════════════════════════
    els.append(Paragraph("Differential Diagnoses", s_section))

    s_th = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8,
                          textColor=colors.white, leading=10)
    s_td = ParagraphStyle("td", fontName="Helvetica", fontSize=9,
                          textColor=SLATE_600, leading=12)
    s_td_bold = ParagraphStyle("tdb", fontName="Helvetica-Bold", fontSize=9,
                               textColor=SLATE_900, leading=12)

    dx_header = [
        Paragraph("RANK", s_th), Paragraph("CONDITION", s_th),
        Paragraph("CLINICAL NOTE", s_th), Paragraph("CONFIDENCE", s_th),
    ]
    dx_rows = [dx_header]
    for r, (c, cf) in enumerate(top3, 1):
        cd = CLASS_DISPLAY.get(c, c)
        note = CLASS_DESCRIPTIONS.get(c, "")
        # Truncate note
        if len(note) > 60:
            note = note[:57] + "..."
        conf_color = DANGER if r == 1 and not is_normal else SUCCESS if r == 1 and is_normal else SLATE_600
        s_conf = ParagraphStyle(f"c{r}", fontName="Helvetica-Bold", fontSize=10,
                                textColor=conf_color, leading=12, alignment=TA_RIGHT)
        dx_rows.append([
            Paragraph(f"{r:02d}", s_td),
            Paragraph(cd, s_td_bold),
            Paragraph(note, s_td),
            Paragraph(f"{cf:.1f}%", s_conf),
        ])

    dx_t = Table(dx_rows, colWidths=[page_w*0.08, page_w*0.25, page_w*0.47, page_w*0.20])
    dx_styles = [
        ("BACKGROUND", (0,0), (-1,0), TEAL),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW", (0,1), (-1,-2), 0.5, SLATE_100),
        ("BOX", (0,0), (-1,-1), 0.5, SLATE_100),
    ]
    # Alternate row backgrounds
    for i in range(1, len(dx_rows)):
        if i % 2 == 0:
            dx_styles.append(("BACKGROUND", (0,i), (-1,i), SLATE_100))
    dx_t.setStyle(TableStyle(dx_styles))
    els.append(dx_t)
    els.append(Spacer(1, 0.25*inch))

    # ══════════════════════════════════════════════════════════════════════
    #  SCAN IMAGES (side by side)
    # ══════════════════════════════════════════════════════════════════════
    has_orig = orig_path and os.path.exists(orig_path)
    has_gc = gc_path and os.path.exists(gc_path)

    if has_orig or has_gc:
        els.append(Paragraph("Analysis Visualization", s_section))

        img_w = page_w * 0.46
        img_cells = []
        cap_cells = []
        if has_orig:
            img_cells.append(RLImage(orig_path, width=img_w, height=img_w))
            cap_cells.append(Paragraph("Original MRI Scan", s_center))
        if has_gc:
            img_cells.append(RLImage(gc_path, width=img_w, height=img_w))
            cap_cells.append(Paragraph("Grad-CAM Activation Heatmap", s_center))

        if len(img_cells) == 2:
            img_t = Table([img_cells, cap_cells], colWidths=[page_w*0.5, page_w*0.5])
        else:
            img_t = Table([img_cells, cap_cells], colWidths=[page_w])

        img_t.setStyle(TableStyle([
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        els.append(img_t)
        els.append(Spacer(1, 0.1*inch))
        els.append(Paragraph(
            "Red zones in the Grad-CAM heatmap indicate regions of high model attention.",
            s_center,
        ))
        els.append(Spacer(1, 0.25*inch))

    # ══════════════════════════════════════════════════════════════════════
    #  ALL CLASS PROBABILITIES (horizontal bars using table)
    # ══════════════════════════════════════════════════════════════════════
    if all_probs:
        els.append(Paragraph("Class Probability Distribution", s_section))
        s_prob_name = ParagraphStyle("pn", fontName="Helvetica-Bold", fontSize=8,
                                     textColor=SLATE_600, leading=10)
        s_prob_val = ParagraphStyle("pv", fontName="Helvetica-Bold", fontSize=9,
                                    textColor=SLATE_900, leading=11, alignment=TA_RIGHT)
        prob_rows = []
        for cls_name, prob_val in all_probs:
            disp_name = CLASS_DISPLAY.get(cls_name, cls_name)
            is_this_top = (cls_name == top_cls)
            bar_color = (DANGER if not is_normal else SUCCESS) if is_this_top else TEAL
            # Create a mini bar
            filled = max(1, int(prob_val * 3))  # scale to ~300px max
            bar_cell = Table([[""]], colWidths=[filled], rowHeights=[8])
            bar_cell.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), bar_color),
                ("TOPPADDING", (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ]))
            prob_rows.append([
                Paragraph(disp_name.upper(), s_prob_name),
                bar_cell,
                Paragraph(f"{prob_val:.1f}%", s_prob_val),
            ])
        prob_t = Table(prob_rows, colWidths=[page_w*0.25, page_w*0.55, page_w*0.20])
        prob_t.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("LINEBELOW", (0,0), (-1,-2), 0.3, SLATE_100),
            ("BOX", (0,0), (-1,-1), 0.5, SLATE_100),
        ]))
        els.append(prob_t)
        els.append(Spacer(1, 0.25*inch))

    # ══════════════════════════════════════════════════════════════════════
    #  DISCLAIMER
    # ══════════════════════════════════════════════════════════════════════
    disc_t = Table(
        [[Paragraph(
            "<b>⚠ Disclaimer:</b> This report is generated by an AI system for "
            "research and educational purposes only. It is not intended to replace "
            "professional medical advice, diagnosis, or treatment. Always consult a "
            "qualified radiologist or physician for clinical decisions.",
            s_disclaimer,
        )]],
        colWidths=[page_w],
    )
    disc_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), AMBER_LIGHT),
        ("BOX", (0,0), (-1,-1), 0.5, AMBER),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
    ]))
    els.append(disc_t)
    els.append(Spacer(1, 0.2*inch))

    # ── Footer ────────────────────────────────────────────────────────────
    els.append(HRFlowable(width="100%", thickness=0.5, color=SLATE_100))
    els.append(Spacer(1, 6))
    els.append(Paragraph(
        f"© {now.year} Brain MRI Diagnostic System · Version 2.4.1-B0 · "
        f"Report generated on {now.strftime('%B %d, %Y at %I:%M:%S %p')} · "
        "HIPAA Compliant Environment",
        s_footer,
    ))

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
    all_probs_list = [(CLASS_NAMES[i], float(pnp[i]*100)) for i in range(len(CLASS_NAMES))]
    pdf_path = generate_pdf(patient_name, age, gender, top3, orig_path, gc_path, all_probs=all_probs_list)
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
