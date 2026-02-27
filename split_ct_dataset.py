import os
import shutil
import random

RAW_DIR = "data/ct/raw"
OUTPUT_DIR = "data/ct"

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

def split_class(class_name):
    class_path = os.path.join(RAW_DIR, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT["train"])
    val_end = train_end + int(total * SPLIT["val"])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        out_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for f in files:
            src = os.path.join(class_path, f)
            dst = os.path.join(out_dir, f)
            shutil.copy(src, dst)

    print(f"✅ {class_name}: {len(images)} images split")

def main():
    for cls in ["no", "yes"]:
        split_class(cls)

    print("\n🎯 CT dataset split completed successfully")

if __name__ == "__main__":
    main()
