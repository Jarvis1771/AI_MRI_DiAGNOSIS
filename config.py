"""
config.py — Single source of truth for the Brain MRI Multi-Disease Classifier.
All scripts import from here; never hard-code class names or paths elsewhere.
"""

# ── Class Labels ─────────────────────────────────────────────────────────────
# Must match the sub-folder names (alphabetical) inside data/mri/train/
CLASS_NAMES = [
    "Alzheimers",
    "BrainTumor",
    "Hemorrhage",
    "MultipleSclerosis",
    "Normal",
    "Stroke",
]

NUM_CLASSES = len(CLASS_NAMES)

# ── Human-readable display names ──────────────────────────────────────────────
CLASS_DISPLAY = {
    "Alzheimers":       "Alzheimer's / Atrophy",
    "BrainTumor":       "Brain Tumor",
    "Hemorrhage":       "Hemorrhage",
    "MultipleSclerosis":"Multiple Sclerosis",
    "Normal":           "Normal Brain",
    "Stroke":           "Stroke (Ischemic)",
}

# Brief clinical note shown per prediction in the PDF report
CLASS_DESCRIPTIONS = {
    "Alzheimers":       "Diffuse cortical atrophy pattern consistent with neurodegenerative disease.",
    "BrainTumor":       "Abnormal mass lesion identified. Urgent neurological review recommended.",
    "Hemorrhage":       "Hyperdense region suggesting intracranial hemorrhage. Emergency evaluation advised.",
    "MultipleSclerosis":"Periventricular white-matter lesions consistent with demyelinating disease.",
    "Normal":           "No significant intracranial pathology detected on this scan.",
    "Stroke":           "Ischemic territory signal change detected. Immediate clinical correlation required.",
}

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/brain_mri_multiclass.pth"
ARCH       = "efficientnet_b0"   # torchvision model name

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224

# ImageNet normalisation (used for EfficientNet pretrained weights)
NORM_MEAN  = [0.485, 0.456, 0.406]
NORM_STD   = [0.229, 0.224, 0.225]

# ── Data directories ──────────────────────────────────────────────────────────
TRAIN_DIR  = "data/mri/train"
VAL_DIR    = "data/mri/val"
