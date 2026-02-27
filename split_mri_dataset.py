import os
import shutil
import random

# ---------------- CONFIG ----------------
RAW_DATA_DIR = "data/mri/raw/brain_tumor_dataset"
OUTPUT_BASE = "data/mri"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
# ----------------------------------------

random.seed(RANDOM_SEED)

classes = ["yes", "no"]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_BASE, split, cls), exist_ok=True)

for cls in classes:
    class_dir = os.path.join(RAW_DATA_DIR, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        for file in files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(OUTPUT_BASE, split, cls, file)
            shutil.copy2(src, dst)

    print(f"✔ {cls.upper()} -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

print("\n✅ MRI dataset successfully split into Train / Val / Test")
