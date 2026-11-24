#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from config import DATA_ROOT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone full-image evaluation for repository checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--image-dir")
    parser.add_argument("--label-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers non-negative")
    return args


def run(args):
    from runtime import load_categories, load_detector, load_preprocessing, resolve_device
    from train_det_fast import EVAL_IOU_THRS, DetDataset, evaluate

    root = Path(args.data_root)
    image_dir = Path(args.image_dir) if args.image_dir else root / "images" / args.split
    label_dir = Path(args.label_dir) if args.label_dir else root / "labels" / args.split
    categories = load_categories(args.classes)
    device = resolve_device(args.device)
    model, ckpt = load_detector(args.weights, categories, device=device)
    preprocessing = load_preprocessing(ckpt)
    image_size = preprocessing["image_size"]
    dataset = DetDataset(
        str(image_dir), str(label_dir), args.classes,
        img_size=image_size, augment=False,
        patch_size=model.patch_size,
        image_mean=preprocessing["mean"], image_std=preprocessing["std"],
        pad_value=preprocessing["pad"],
    )
    results, class_details, extra = evaluate(
        model, dataset, device,
        eval_batch=args.batch_size, eval_workers=args.workers,
    )
    model.eval()
    payload = {
        "results": results,
        "class_details": {str(key): value for key, value in class_details.items()},
        "evaluation": extra,
        "protocol": {
            "iou_thresholds": list(EVAL_IOU_THRS),
            "confidence_threshold": 0.05,
            "nms_iou": 0.5,
            "image_size": image_size,
            "resize_mode": "full-image aspect-preserving letterbox",
            "autocast": "bf16" if str(device).startswith("cuda") else "fp32",
            "ap_integration": "all-point precision-recall; not COCO 101-point",
            "max_detections_per_class_per_image": 500,
            "checkpoint_epoch": ckpt.get("epoch"),
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
