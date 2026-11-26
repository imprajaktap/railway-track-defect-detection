
from pathlib import Path
from collections import Counter
from PIL import Image
import yaml

DATASET = Path(
    r"C:\Users\Prajakta Palkar\Downloads\RSD-YOLO-Dataset-v1.0\RSD-YOLO-Dataset-v1.0"
)

CLASSES = {
    0: "spalling",
    1: "wheel burn",
    2: "squat"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 55)
print("RAILWAY DEFECT DATASET VALIDATION")
print("=" * 55)

# Read configuration
yaml_path = DATASET / "data.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("\nDATASET CONFIGURATION")
print(config)

class_counts = Counter()
total_images = 0
total_objects = 0
missing_labels = []
invalid_labels = []
unreadable_images = []
empty_labels = []

for split in ["train", "val", "test"]:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = {p.stem: p for p in label_dir.rglob("*.txt")}

    split_objects = 0

    for image_path in images:

        total_images += 1

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            unreadable_images.append(str(image_path))
            continue

        label_path = labels.get(image_path.stem)

        if label_path is None:
            missing_labels.append(str(image_path))
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()

        if not lines:
            empty_labels.append(str(label_path))
            continue

        for line in lines:
            values = line.split()

            try:
                if len(values) != 5:
                    raise ValueError("Expected 5 values")

                class_id = int(values[0])
                coords = [float(v) for v in values[1:]]

                if class_id not in CLASSES:
                    raise ValueError("Unknown class ID")

                if not all(0 <= v <= 1 for v in coords):
                    raise ValueError("Coordinates outside 0-1")

                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError("Invalid box dimensions")

                class_counts[class_id] += 1
                total_objects += 1
                split_objects += 1

            except Exception as error:
                invalid_labels.append(
                    (str(label_path), line, str(error))
                )

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Valid objects:", split_objects)

print("\n" + "=" * 55)
print("VALIDATION SUMMARY")
print("=" * 55)

print("Total images:", total_images)
print("Total valid objects:", total_objects)
print("Missing labels:", len(missing_labels))
print("Unreadable images:", len(unreadable_images))
print("Empty labels:", len(empty_labels))
print("Invalid annotations:", len(invalid_labels))

print("\nCLASS DISTRIBUTION")

for class_id, name in CLASSES.items():
    print(name, ":", class_counts[class_id])

print("\nVALIDATION FINISHED")
