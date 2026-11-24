#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CONF_THR = 0.05
NMS_THR = 0.5
MAX_DETECTIONS = 500


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Full-image inference for the LingBot-Vision detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Repository .pth checkpoint")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--classes", required=True, help="classes.txt in checkpoint order")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return args


def _visualize(image, detections, categories, output_path):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for score, class_id, box in detections:
        draw.rectangle(box, outline=(255, 40, 40), width=2)
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{categories[class_id]} {score:.2f}", fill=(255, 255, 0),
        )
    image.save(output_path, quality=92)


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from runtime import (
        decode_predictions, load_categories, load_detector, load_preprocessing, prepare_image,
    )

    source = Path(args.source)
    if source.is_dir():
        files = sorted(
            path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        files = [source]
    else:
        raise FileNotFoundError(f"no supported image found: {source}")
    if not files:
        raise ValueError(f"no images found in {source}")

    categories = load_categories(args.classes)
    model, ckpt = load_detector(args.weights, categories, device=args.device)
    preprocessing = load_preprocessing(ckpt)
    image_size = preprocessing["image_size"]
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    if not args.no_visualize:
        visualization_dir.mkdir(exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    image_count = 0
    detection_count = 0
    mean = torch.tensor(preprocessing["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(preprocessing["std"], device=device).view(1, 3, 1, 1)
    with prediction_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(files), args.batch_size):
            group = files[start:start + args.batch_size]
            images, arrays, masks, sizes = [], [], [], []
            for image_path in group:
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
                canvas, valid_mask, _ = prepare_image(
                    image, image_size, model.patch_size, preprocessing["pad"],
                )
                images.append(image)
                sizes.append(image.size)
                arrays.append(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1).copy())
                masks.append(valid_mask.numpy())
            pixel_values = torch.from_numpy(np.stack(arrays)).to(device)
            valid_mask = torch.from_numpy(np.stack(masks)).to(device)
            normalized = (pixel_values.float() / 255.0 - mean) / std
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp,
            ):
                reg_pred, _, cls_pred, centerness_pred, _, _ = model(normalized)
            reg_pred = [value.float() for value in reg_pred]
            cls_pred = [value.float() for value in cls_pred]
            centerness_pred = [value.float() for value in centerness_pred]
            for index, (image_path, image, (width, height)) in enumerate(zip(group, images, sizes)):
                decoded = decode_predictions(
                    [value[index] for value in reg_pred],
                    [value[index] for value in cls_pred],
                    [value[index] for value in centerness_pred],
                    valid_mask[index], (width, height), len(categories),
                    conf_threshold=CONF_THR, nms_threshold=NMS_THR,
                    max_detections=MAX_DETECTIONS,
                )
                detections = []
                for class_id, entries in enumerate(decoded):
                    for normalized_box, score in entries:
                        box = (normalized_box * torch.tensor(
                            [width, height, width, height], device=valid_mask.device,
                        )).cpu().tolist()
                        detections.append((float(score), class_id, box))
                detections.sort(key=lambda entry: -entry[0])
                for score, class_id, box in detections:
                    x1, y1, x2, y2 = box
                    record = {
                        "image": image_path.name, "class_id": class_id,
                        "class_name": categories[class_id], "score": round(score, 6),
                        "bbox_xyxy": [round(value, 3) for value in box],
                        "bbox_xywh": [
                            round(x1, 3), round(y1, 3),
                            round(x2 - x1, 3), round(y2 - y1, 3),
                        ],
                        "bbox_norm": [
                            round(x1 / width, 6), round(y1 / height, 6),
                            round(x2 / width, 6), round(y2 / height, 6),
                        ],
                        "image_width": width, "image_height": height,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                if not args.no_visualize:
                    _visualize(
                        image.copy(), detections, categories,
                        visualization_dir / f"{image_path.stem}.jpg",
                    )
                image_count += 1
                detection_count += len(detections)
                print(f"[infer] {image_path.name}: {len(detections)} detections")
    print(
        f"[infer] {image_count} images, {detection_count} detections -> {prediction_path}"
    )
    return 0


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
