"""
prepare_dataset.py — Organise downloaded Kaggle MRI datasets into the
                     6-class folder structure expected by train_brain_mri.py.

Supports:
  1. Brain Tumor MRI Dataset  (masoudnickparvar/brain-tumor-mri-dataset)
  2. Alzheimer's MRI Dataset  (tourist55/alzheimers-dataset)
  3. Multiple Sclerosis MRI   (buraktaci/multiple-sclerosis)
  4. Any extra folder you point at for stroke / hemorrhage images

Usage:
  # Step 1 — Download datasets from Kaggle (run in terminal):
  #   kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset -p data/raw/
  #   kaggle datasets download tourist55/alzheimers-dataset              -p data/raw/
  #   kaggle datasets download buraktaci/multiple-sclerosis              -p data/raw/
  #   (then unzip all inside data/raw/ OR let this script auto-unzip)

  # Step 2 — Run this script:
  python3 prepare_dataset.py

  # Step 3 — Optionally add extra stroke / hemorrhage images:
  python3 prepare_dataset.py --stroke path/to/stroke/folder
  python3 prepare_dataset.py --hemorrhage path/to/hemorrhage/folder

Output:
  data/mri/train/{Alzheimers, BrainTumor, Hemorrhage, MultipleSclerosis, Normal, Stroke}/
  data/mri/val/  {same 6 folders}   (80/20 split)
"""

import os
import sys
import shutil
import random
import zipfile
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
TRAIN_DIR = Path("data/mri/train")
VAL_DIR   = Path("data/mri/val")
VAL_SPLIT = 0.20
SEED      = 42
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_image(path):
    return path.suffix.lower() in IMG_EXTS


def collect_images(src_dir):
    """Recursively collect all image paths from a directory."""
    return [p for p in Path(src_dir).rglob("*") if p.is_file() and is_image(p)]


def unzip_all():
    """Auto-unzip any .zip files in data/raw/."""
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        return
    print(f"\n📦 Found {len(zips)} zip file(s) — extracting...")
    for z in zips:
        dest = RAW_DIR / z.stem
        if not dest.exists():
            print(f"   Extracting {z.name}...")
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dest)
            print(f"   ✅ Extracted → {dest}")
        else:
            print(f"   ⏭️  {z.name} already extracted — skipping")


def copy_split(images, class_name, source_label=""):
    """Shuffle images, split 80/20, copy to train/val folders."""
    if not images:
        return 0, 0

    images = list(set(images))  # deduplicate
    random.shuffle(images)
    n_val   = max(1, int(len(images) * VAL_SPLIT))
    val_imgs = images[:n_val]
    trn_imgs = images[n_val:]

    for img_list, dest_root in [(trn_imgs, TRAIN_DIR), (val_imgs, VAL_DIR)]:
        dest = dest_root / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for p in img_list:
            target = dest / p.name
            if target.exists():
                target = dest / f"{p.parent.name}_{p.name}"
            shutil.copy2(p, target)

    tag = f"  ({source_label})" if source_label else ""
    print(f"   {class_name:<22} train: {trn_imgs.__len__():>5}   val: {val_imgs.__len__():>5}{tag}")
    return len(trn_imgs), len(val_imgs)


# ── Dataset Processors ────────────────────────────────────────────────────────

def process_brain_tumor(totals):
    """Brain Tumor MRI Dataset by masoudnickparvar.
    Folder structure: Training/{glioma,meningioma,pituitary,notumor}
                      Testing/ {same}
    """
    training_dirs = list(RAW_DIR.rglob("Training"))
    testing_dirs  = list(RAW_DIR.rglob("Testing"))

    if not training_dirs and not testing_dirs:
        print("   ⚠️  Brain Tumor dataset not found — skipping")
        return

    all_tumor, all_normal = [], []
    for base in training_dirs + testing_dirs:
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            imgs = collect_images(sub)
            name = sub.name.lower()
            if name in ("glioma", "meningioma", "pituitary",
                        "glioma_tumor", "meningioma_tumor", "pituitary_tumor"):
                all_tumor += imgs
            elif name in ("notumor", "no_tumor", "normal"):
                all_normal += imgs

    if all_tumor:
        t, v = copy_split(all_tumor, "BrainTumor", "Brain Tumor Dataset")
        totals["BrainTumor"] = (totals["BrainTumor"][0]+t, totals["BrainTumor"][1]+v)
    if all_normal:
        t, v = copy_split(all_normal, "Normal", "Brain Tumor Dataset — no tumor")
        totals["Normal"] = (totals["Normal"][0]+t, totals["Normal"][1]+v)


def process_alzheimers(totals):
    """Alzheimer's Dataset by tourist55.
    Merges all Demented variants → Alzheimers, NonDemented → Normal.
    """
    demented, non_demented = [], []

    for p in RAW_DIR.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name.lower().replace(" ", "").replace("_", "")
        if name in ("milddemented", "moderatedemented", "verymilddemented",
                     "mild_demented", "moderate_demented", "very_mild_demented"):
            demented += collect_images(p)
        elif name in ("nondemented", "non_demented"):
            non_demented += collect_images(p)

    if demented:
        t, v = copy_split(demented, "Alzheimers", "Alzheimers — demented")
        totals["Alzheimers"] = (totals["Alzheimers"][0]+t, totals["Alzheimers"][1]+v)
    if non_demented:
        t, v = copy_split(non_demented, "Normal", "Alzheimers — non-demented")
        totals["Normal"] = (totals["Normal"][0]+t, totals["Normal"][1]+v)

    if not demented and not non_demented:
        print("   ⚠️  Alzheimer's dataset not found — skipping")


def process_ms(totals):
    """Multiple Sclerosis dataset (buraktaci or similar)."""
    keywords = ["ms", "multiple_sclerosis", "ms_lesion", "multiple sclerosis"]
    all_ms = []

    for p in RAW_DIR.rglob("*"):
        if p.is_dir() and p.name.lower() in keywords:
            all_ms += collect_images(p)

    all_ms = list(set(all_ms))
    if all_ms:
        t, v = copy_split(all_ms, "MultipleSclerosis", "MS dataset")
        totals["MultipleSclerosis"] = (totals["MultipleSclerosis"][0]+t,
                                       totals["MultipleSclerosis"][1]+v)
    else:
        print("   ⚠️  Multiple Sclerosis dataset not found — skipping")


def process_stroke(totals, override_folder=None):
    """Brain Stroke CT Dataset by afridirahman.
    Structure: brain-stroke/Brain_Data_Organised/{Normal, Stroke}
    """
    all_stroke = []

    if override_folder:
        imgs = collect_images(Path(override_folder))
        all_stroke += imgs
    else:
        # Auto-detect from downloaded dataset
        for p in RAW_DIR.rglob("*"):
            if p.is_dir() and p.name.lower() == "stroke":
                all_stroke += collect_images(p)

    all_stroke = list(set(all_stroke))
    if all_stroke:
        t, v = copy_split(all_stroke, "Stroke", "Stroke dataset")
        totals["Stroke"] = (totals["Stroke"][0]+t, totals["Stroke"][1]+v)
    else:
        print("   ⚠️  Stroke dataset not found — skipping")


def process_hemorrhage(totals, override_folder=None):
    """Brain CT Hemorrhage Dataset by abdulkader90.
    Structure: brain-hemorrhage/Data/Hemorrhagic/KANAMA/
    """
    all_hem = []

    if override_folder:
        imgs = collect_images(Path(override_folder))
        all_hem += imgs
    else:
        # Auto-detect: look for 'hemorrhagic', 'kanama', 'hemorrhage' folders
        keywords = ["hemorrhagic", "kanama", "hemorrhage"]
        for p in RAW_DIR.rglob("*"):
            if p.is_dir() and p.name.lower() in keywords:
                all_hem += collect_images(p)

    all_hem = list(set(all_hem))
    if all_hem:
        t, v = copy_split(all_hem, "Hemorrhage", "Hemorrhage dataset")
        totals["Hemorrhage"] = (totals["Hemorrhage"][0]+t, totals["Hemorrhage"][1]+v)
    else:
        print("   ⚠️  Hemorrhage dataset not found — skipping")


def process_extra(folder, class_name, totals):
    """Copy images from a user-provided folder into a target class."""
    if not folder:
        return
    p = Path(folder)
    if not p.exists():
        print(f"   ⚠️  {class_name} folder not found: {folder}")
        return
    imgs = collect_images(p)
    if imgs:
        t, v = copy_split(imgs, class_name, str(folder))
        totals[class_name] = (totals[class_name][0]+t, totals[class_name][1]+v)
    else:
        print(f"   ⚠️  No images found in {folder}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Organise Kaggle MRI datasets into 6-class folders")
    parser.add_argument("--stroke",     type=str, default=None, help="Path to folder with stroke MRI images")
    parser.add_argument("--hemorrhage", type=str, default=None, help="Path to folder with hemorrhage MRI images")
    parser.add_argument("--clean",      action="store_true",    help="Delete existing train/val folders before processing")
    args = parser.parse_args()

    print("=" * 60)
    print("  Brain MRI Multi-Disease — Dataset Preparation")
    print("=" * 60)

    # Validate raw dir
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not any(RAW_DIR.iterdir()):
        print(f"\n❌ data/raw/ is empty! Download datasets first:")
        print(f"   kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset -p data/raw/")
        print(f"   kaggle datasets download tourist55/alzheimers-dataset -p data/raw/")
        print(f"   kaggle datasets download buraktaci/multiple-sclerosis -p data/raw/")
        sys.exit(1)

    # Optionally clean existing data
    if args.clean:
        print("\n🗑️  Cleaning existing train/val folders...")
        for cls in ["Alzheimers", "BrainTumor", "Hemorrhage",
                     "MultipleSclerosis", "Normal", "Stroke"]:
            for d in [TRAIN_DIR / cls, VAL_DIR / cls]:
                if d.exists():
                    shutil.rmtree(d)
        print("   ✅ Cleaned")

    # Ensure output folders exist
    for cls in ["Alzheimers", "BrainTumor", "Hemorrhage",
                 "MultipleSclerosis", "Normal", "Stroke"]:
        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)
        (VAL_DIR   / cls).mkdir(parents=True, exist_ok=True)

    # Auto-unzip
    unzip_all()

    # Process each dataset
    totals = {c: (0, 0) for c in ["Alzheimers", "BrainTumor", "Hemorrhage",
                                    "MultipleSclerosis", "Normal", "Stroke"]}

    print("\n🔍 Processing Brain Tumor dataset...")
    process_brain_tumor(totals)

    print("\n🔍 Processing Alzheimer's dataset...")
    process_alzheimers(totals)

    print("\n🔍 Processing Multiple Sclerosis dataset...")
    process_ms(totals)

    print("\n🔍 Processing Stroke images...")
    process_stroke(totals, override_folder=args.stroke)

    print("\n🔍 Processing Hemorrhage images...")
    process_hemorrhage(totals, override_folder=args.hemorrhage)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 DATASET SUMMARY")
    print("=" * 60)
    print(f"  {'Class':<22} {'Train':>8}  {'Val':>8}  {'Total':>8}")
    print("  " + "-" * 50)

    grand_train, grand_val = 0, 0
    empty_classes = []
    for cls, (t, v) in sorted(totals.items()):
        total = t + v
        print(f"  {cls:<22} {t:>8}  {v:>8}  {total:>8}")
        grand_train += t
        grand_val   += v
        if total == 0:
            empty_classes.append(cls)

    print("  " + "-" * 50)
    print(f"  {'TOTAL':<22} {grand_train:>8}  {grand_val:>8}  {grand_train+grand_val:>8}")

    if empty_classes:
        print(f"\n  ⚠️  Empty classes: {', '.join(empty_classes)}")
        print(f"     Add data for these classes before training.")
        print(f"     Training will still work — WeightedRandomSampler handles imbalance.")

    print(f"\n  ✅ Dataset prepared in data/mri/train/ and data/mri/val/")
    print(f"\n  Next step:  python3 train_brain_mri.py --check-only")
    print(f"              python3 train_brain_mri.py")
    print()


if __name__ == "__main__":
    main()
