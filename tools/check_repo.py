#!/usr/bin/env python3
"""Check repository structure and the bundled SODA-A samples without weights."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGES = 10
EXPECTED_CLASSES = 9
EXPECTED_SPLITS = {"train": 8, "val": 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".onnx", ".safetensors", ".ckpt", ".bin", ".log", ".pyc",
}
LOCAL_PATH_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/(?:Users|home)/)[^\s\"']+")


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def check_samples() -> dict[str, int]:
    sample_root = ROOT / "data_samples"
    images = sorted(
        p for p in (sample_root / "images").rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted((sample_root / "labels").rglob("*.txt"))
    if len(images) != EXPECTED_IMAGES:
        fail(f"expected {EXPECTED_IMAGES} sample images, found {len(images)}")
    if len(labels) != EXPECTED_IMAGES:
        fail(f"expected {EXPECTED_IMAGES} sample labels, found {len(labels)}")

    classes = [
        line.strip() for line in (sample_root / "classes.txt").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    if len(classes) != EXPECTED_CLASSES:
        fail(f"expected {EXPECTED_CLASSES} classes, found {len(classes)}")

    total_instances = 0
    covered_classes: set[int] = set()
    split_counts = {split: 0 for split in EXPECTED_SPLITS}
    for image_path in images:
        relative_path = image_path.relative_to(sample_root)
        if (
            len(relative_path.parts) != 3
            or relative_path.parts[0] != "images"
            or relative_path.parts[1] not in EXPECTED_SPLITS
        ):
            fail(f"image is outside the YOLO train/val layout: {relative_path.as_posix()}")
        split_counts[relative_path.parts[1]] += 1
        label_relative = Path(*relative_path.parts[1:]).with_suffix(".txt")
        label_path = sample_root / "labels" / label_relative
        if not label_path.is_file():
            fail(f"missing label for {relative_path.as_posix()}")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            fail(f"invalid image size for {image_path.name}: {width}x{height}")

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                fail(
                    f"{label_path.name}:{line_number}: expected 5 YOLO fields, "
                    f"found {len(fields)}"
                )
            class_id = int(fields[0])
            cx, cy, box_w, box_h = map(float, fields[1:])
            if not 0 <= class_id < len(classes):
                fail(f"{label_path.name}:{line_number}: class id out of range")
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                fail(f"{label_path.name}:{line_number}: center out of range")
            if not (0.0 < box_w <= 1.0 and 0.0 < box_h <= 1.0):
                fail(f"{label_path.name}:{line_number}: invalid box size")
            if cx - box_w / 2 < -1e-6 or cx + box_w / 2 > 1.0 + 1e-6:
                fail(f"{label_path.name}:{line_number}: box exceeds image width")
            if cy - box_h / 2 < -1e-6 or cy + box_h / 2 > 1.0 + 1e-6:
                fail(f"{label_path.name}:{line_number}: box exceeds image height")
            total_instances += 1
            covered_classes.add(class_id)

    if covered_classes != set(range(len(classes))):
        fail(f"samples do not cover all classes: {sorted(covered_classes)}")
    if split_counts != EXPECTED_SPLITS:
        fail(f"expected YOLO split counts {EXPECTED_SPLITS}, found {split_counts}")

    manifest_path = sample_root / "manifest.csv"
    if not manifest_path.is_file():
        fail("missing data_samples/manifest.csv")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_IMAGES:
        fail(f"manifest has {len(rows)} rows, expected {EXPECTED_IMAGES}")
    manifest_images = {row["image"] for row in rows}
    expected_images = {p.relative_to(sample_root).as_posix() for p in images}
    if manifest_images != expected_images:
        fail("manifest image list does not match bundled images")
    for row in rows:
        image_path = sample_root / row["image"]
        if row["split"] != Path(row["image"]).parts[1]:
            fail(f"manifest split does not match image path: {row['image']}")
        expected_label = str(Path("labels") / Path(*Path(row["image"]).parts[1:]).with_suffix(".txt"))
        if Path(row["label"]).as_posix() != Path(expected_label).as_posix():
            fail(f"manifest label does not match image path: {row['image']}")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            fail(f"SHA-256 mismatch for {row['image']}")
        if int(row["instances"]) <= 0:
            fail(f"sample {row['image']} has no instances")

    return {
        "images": len(images),
        "labels": len(labels),
        "classes": len(classes),
        "instances": total_instances,
    }


def check_repository_files() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "requirements.txt",
        ROOT / ".gitignore",
        ROOT / "NOTICE.md",
        ROOT / "DATA_NOTICE.md",
        ROOT / "model.py",
        ROOT / "config.py",
        ROOT / "train_det_fast.py",
        ROOT / "runtime.py",
        ROOT / "evaluate.py",
        ROOT / "infer.py",
        ROOT / "third_party/lingbot_vision/LICENSE",
        ROOT / "third_party/lingbot_vision/README.md",
        ROOT / "third_party/lingbot_vision/SOURCE.md",
    ]
    for path in required:
        if not path.is_file():
            fail(f"missing required file: {path.relative_to(ROOT)}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"forbidden artifact in repository: {path.relative_to(ROOT)}")
        if path.suffix.lower() not in {".py", ".md", ".txt", ".yaml", ".yml", ".toml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        match = LOCAL_PATH_RE.search(text)
        if match:
            fail(
                f"local absolute path in {path.relative_to(ROOT)}: "
                f"{match.group(0)}"
            )


def main() -> int:
    stats = check_samples()
    check_repository_files()
    print(
        "OK: "
        f"{stats['images']} images, {stats['labels']} labels, "
        f"{stats['classes']} classes, {stats['instances']} instances; "
        "manifest checksums and repository paths are valid."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
