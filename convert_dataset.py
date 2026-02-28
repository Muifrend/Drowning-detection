import json
import os
from pathlib import Path
from PIL import Image

CLASSES = {0: "swimming", 1: "drowning"}

TRAIN_IMAGES = Path("datasets/train/images")
TRAIN_LABELS = Path("datasets/train/labels")
VAL_IMAGES   = Path("datasets/valid/images")
VAL_LABELS   = Path("datasets/valid/labels")

def convert_label_file(label_path, image_path):
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return None

    detections = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls  = int(parts[0])
            cx   = float(parts[1])
            cy   = float(parts[2])
            w    = float(parts[3])
            h    = float(parts[4])

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # clamp to 0-1
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            # scale to 1024
            loc_y1 = int(y1 * 1024)
            loc_x1 = int(x1 * 1024)
            loc_y2 = int(y2 * 1024)
            loc_x2 = int(x2 * 1024)

            label = CLASSES.get(cls, "unknown")
            detections.append(
                f"<loc{loc_y1:04d}><loc{loc_x1:04d}>"
                f"<loc{loc_y2:04d}><loc{loc_x2:04d}> {label}"
            )

    if not detections:
        return None

    return " ; ".join(detections)

def build_split(images_dir, labels_dir, split_name):
    samples = []
    class_counts = {"swimming": 0, "drowning": 0}
    skipped = 0

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(images_dir.glob("*.png"))

    for img_path in image_files:
        label_path = labels_dir / img_path.with_suffix(".txt").name

        if not label_path.exists():
            skipped += 1
            continue

        suffix = convert_label_file(label_path, img_path)
        if suffix is None:
            skipped += 1
            continue

        for cls_name in class_counts:
            if cls_name in suffix:
                class_counts[cls_name] += suffix.count(cls_name)

        samples.append({
            "image": str(img_path),
            "prefix": "detect swimming ; drowning",
            "suffix": suffix
        })

    print(f"{split_name}: {len(samples)} samples, {skipped} skipped")
    print(f"  Class counts: {class_counts}")
    return samples

def main():
    print("Converting dataset to PaliGemma format...\n")

    train_samples = build_split(TRAIN_IMAGES, TRAIN_LABELS, "Train")
    val_samples   = build_split(VAL_IMAGES,   VAL_LABELS,   "Val")

    with open("paligemma_train.json", "w") as f:
        json.dump(train_samples, f, indent=2)

    with open("paligemma_val.json", "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"\nTotal train: {len(train_samples)}")
    print(f"Total val:   {len(val_samples)}")
    print("\nSaved paligemma_train.json and paligemma_val.json")
    print("Done.")

if __name__ == "__main__":
    main()
