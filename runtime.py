import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms


def resolve_device(value="auto"):
    if value in (None, "auto"):
        value = "cuda" if torch.cuda.is_available() else "cpu"
    value = str(value)
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return value


def load_categories(path):
    categories = [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not categories:
        raise ValueError(f"empty class file: {path}")
    if len(set(categories)) != len(categories):
        raise ValueError(f"duplicate class names in {path}")
    return categories


def prepare_image(image, image_size, patch_size, pad_value):
    if type(image_size) is not int or image_size <= 0 or image_size % patch_size:
        raise ValueError("image_size must be a positive multiple of patch_size")
    if type(pad_value) is not int or not 0 <= pad_value <= 255:
        raise ValueError("pad_value must be an integer in [0, 255]")
    width, height = image.size
    scale = min(1.0, image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if (resized_width, resized_height) != image.size:
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (pad_value,) * 3)
    canvas.paste(image, (0, 0))
    feature_size = image_size // patch_size
    columns = max(1, min(feature_size, math.ceil(resized_width / patch_size - 1e-6)))
    rows = max(1, min(feature_size, math.ceil(resized_height / patch_size - 1e-6)))
    valid_mask = torch.zeros(feature_size, feature_size, dtype=torch.bool)
    valid_mask[:rows, :columns] = True
    return canvas, valid_mask, (resized_width, resized_height)


def decode_predictions(reg_pred, cls_pred, centerness_pred, valid_mask, original_size,
                       num_classes, conf_threshold=0.05, nms_threshold=0.5,
                       max_detections=500):
    import torch.nn.functional as F

    original_width, original_height = original_size
    pixel_scale = torch.tensor(
        [original_width, original_height, original_width, original_height],
        dtype=torch.float32, device=valid_mask.device,
    )
    all_boxes, all_scores = [], []
    for scale in range(2):
        grid_height, grid_width = reg_pred[scale].shape[-2:]
        scale_mask = F.interpolate(
            valid_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(grid_height, grid_width), mode="nearest",
        )[0, 0].bool()
        actual_height = max(1, int(scale_mask.any(dim=1).sum()))
        actual_width = max(1, int(scale_mask.any(dim=0).sum()))
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_height, device=valid_mask.device),
            torch.arange(grid_width, device=valid_mask.device), indexing="ij",
        )
        norm_x = (grid_x.float() + 0.5) / actual_width
        norm_y = (grid_y.float() + 0.5) / actual_height
        reg = reg_pred[scale]
        x1 = (norm_x - reg[0] / actual_width).clamp(0, 1)
        y1 = (norm_y - reg[1] / actual_height).clamp(0, 1)
        x2 = (norm_x + reg[2] / actual_width).clamp(0, 1)
        y2 = (norm_y + reg[3] / actual_height).clamp(0, 1)
        score = torch.sigmoid(cls_pred[scale]) * centerness_pred[scale]
        score[:, ~scale_mask] = 0
        all_boxes.append(torch.stack((x1, y1, x2, y2)).reshape(4, -1).transpose(0, 1))
        all_scores.append(score.reshape(num_classes, -1))
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores, dim=1)
    results = [[] for _ in range(num_classes)]
    for class_id in range(num_classes):
        selected = scores[class_id] > conf_threshold
        if not selected.any():
            continue
        class_boxes = boxes[selected]
        class_scores = scores[class_id][selected]
        keep = nms(class_boxes * pixel_scale, class_scores, nms_threshold)[:max_detections]
        results[class_id].extend(zip(class_boxes[keep], class_scores[keep]))
    return results


def _require_value(metadata, key, expected):
    value = metadata.get(key)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"checkpoint {key} must be {expected!r}, got {value!r}")


def load_preprocessing(ckpt):
    saved = ckpt.get("preprocessing")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint preprocessing metadata must be a dictionary")
    vectors = {}
    for key in ("mean", "std"):
        value = saved.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"checkpoint {key} must contain three numbers")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
            raise ValueError(f"checkpoint {key} must contain three numbers")
        vectors[key] = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in (*vectors["mean"], *vectors["std"])):
        raise ValueError("checkpoint mean/std must be finite")
    if any(v <= 0 for v in vectors["std"]):
        raise ValueError("checkpoint std values must be positive")
    pad = saved.get("pad")
    if type(pad) is not int or not 0 <= pad <= 255:
        raise ValueError("checkpoint pad must be an integer in [0, 255]")
    image_size = saved.get("image_size")
    if type(image_size) is not int or image_size <= 0 or image_size % 16:
        raise ValueError("checkpoint image_size must be a positive multiple of 16")
    return {
        **vectors, "pad": pad, "image_size": image_size,
    }


def load_detector(weights, categories, *, device="auto"):
    if not Path(weights).is_file():
        raise FileNotFoundError(weights)
    device = resolve_device(device)
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dictionary")

    saved_categories = ckpt.get("categories")
    if not isinstance(saved_categories, list) or not saved_categories or not all(
        isinstance(name, str) and name for name in saved_categories
    ):
        raise ValueError("checkpoint categories must be a nonempty list of class names")
    if len(set(saved_categories)) != len(saved_categories):
        raise ValueError("checkpoint categories must be unique")
    if saved_categories != list(categories):
        raise ValueError("checkpoint categories must match --classes order")

    arch = ckpt.get("architecture")
    if not isinstance(arch, dict):
        raise ValueError("checkpoint architecture metadata must be a dictionary")
    num_classes = arch.get("num_classes")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("checkpoint num_classes must be a positive integer")
    if num_classes != len(categories):
        raise ValueError("checkpoint num_classes does not match classes.txt")
    expected_architecture = {
        "variant": "small", "embed_dim": 8,
        "num_inter_layers": 4, "reg_max": 8, "use_p2": True,
    }
    for key, expected in expected_architecture.items():
        _require_value(arch, key, expected)
    preprocessing = load_preprocessing(ckpt)
    _require_value(arch, "img_size", preprocessing["image_size"])

    state_keys = ("backbone", "fpn", "det_head", "p2_conv", "p2_head", "p2_stem")
    for key in state_keys:
        if not isinstance(ckpt.get(key), dict) or not ckpt[key]:
            raise ValueError(f"checkpoint must contain a complete {key} state dictionary")

    from model import LbotDetector
    from train_det_fast import _load_model_weights

    model = LbotDetector(
        variant="small", num_classes=num_classes, embed_dim=8, unfreeze_blocks=0,
        img_size=arch["img_size"], num_inter_layers=4,
        device=device, dtype=torch.float32, backbone_state_dict=ckpt["backbone"],
    ).to(device)
    _load_model_weights(model, ckpt, expected_categories=categories)
    model.eval()
    return model, ckpt
