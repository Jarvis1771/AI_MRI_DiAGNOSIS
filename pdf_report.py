"""
pdf_report.py — Generate a rich multi-class AI Medical Report PDF.
Shows Top-3 differential diagnoses with clinical notes.
"""

import os
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4


def generate_medical_report(
    image_path,
    prediction,
    confidence,
    heatmap_path,
    output_pdf,
    top3=None,              # list of (class_name, confidence_pct) tuples
    class_display=None,     # dict: internal_name → display_name
    class_descriptions=None # dict: internal_name → clinical note
):
    """
    Generate a professional PDF diagnostic report.

    Parameters
    ----------
    image_path           : path to uploaded MRI scan
    prediction           : top-1 class name (internal key)
    confidence           : top-1 confidence as float (0-100)
    heatmap_path         : path to Grad-CAM heatmap (or None)
    output_pdf           : output PDF file path
    top3                 : optional list of (class, conf%) for differential diagnoses
    class_display        : optional display name mapping
    class_descriptions   : optional clinical note mapping
    """
    confidence   = float(confidence)
    doc          = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles       = getSampleStyleSheet()
    els          = []

    # ── Header ──────────────────────────────────────────────────────────────
    els.append(Paragraph("<b>AI Brain MRI Diagnostic Report</b>", styles["Title"]))
    els.append(Spacer(1, 0.1 * inch))
    els.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"]
    ))
    els.append(Spacer(1, 0.3 * inch))

    # ── Primary Result ───────────────────────────────────────────────────────
    display_name = (class_display or {}).get(prediction, prediction)
    clinical_note= (class_descriptions or {}).get(prediction, "")

    summary_data = [
        ["Primary Diagnosis", display_name],
        ["Confidence",        f"{confidence:.1f}%"],
        ["Clinical Note",     clinical_note or "N/A"],
    ]
    summary_table = Table(summary_data, colWidths=[140, 360])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1),  colors.HexColor("#2D6A9F")),
        ("TEXTCOLOR",   (0, 0), (0, -1),  colors.white),
        ("FONTNAME",    (0, 0), (0, -1),  "Helvetica-Bold"),
        ("BACKGROUND",  (1, 0), (1, 0),   colors.HexColor("#EBF5FB")),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
    ]))
    els.append(summary_table)
    els.append(Spacer(1, 0.4 * inch))

    # ── Top-3 Differential Diagnoses ─────────────────────────────────────────
    if top3:
        els.append(Paragraph("<b>Top-3 Differential Diagnoses</b>", styles["Heading2"]))
        dx_data = [["Rank", "Condition", "Confidence"]]
        for rank, (cls, conf) in enumerate(top3, 1):
            dx_data.append([
                f"#{rank}",
                (class_display or {}).get(cls, cls),
                f"{conf:.1f}%"
            ])
        dx_table = Table(dx_data, colWidths=[50, 280, 100])
        dx_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),  colors.HexColor("#2D6A9F")),
            ("TEXTCOLOR",  (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, 1),  colors.HexColor("#EBF5FB")),
        ]))
        els.append(dx_table)
        els.append(Spacer(1, 0.4 * inch))

    # ── Scan Images ──────────────────────────────────────────────────────────
    if image_path and os.path.exists(image_path):
        els.append(Paragraph("<b>Uploaded MRI Scan</b>", styles["Heading3"]))
        els.append(Spacer(1, 0.15 * inch))
        els.append(Image(image_path, width=3.5 * inch, height=3.5 * inch))
        els.append(Spacer(1, 0.3 * inch))

    if heatmap_path and os.path.exists(heatmap_path):
        els.append(Paragraph("<b>Grad-CAM Attention Map</b>", styles["Heading3"]))
        els.append(Spacer(1, 0.15 * inch))
        els.append(Image(heatmap_path, width=3.5 * inch, height=3.5 * inch))
        els.append(Spacer(1, 0.3 * inch))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    els.append(Paragraph(
        "<i>⚠️ Disclaimer: This AI-generated report is for research and educational "
        "purposes only. It is not a substitute for diagnosis by a qualified medical "
        "professional. Always consult a licensed radiologist or physician.</i>",
        styles["Normal"]
    ))

    doc.build(els)
