import argparse
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms as _torchvision_nms
from tqdm import tqdm

from config import (
    BACKBONE_LR,
    BATCH_SIZE,
    CLS_IOU_AWARE,
    CLS_IOU_FLOOR,
    CLS_IOU_POWER,
    CONF_THRESHOLD,
    DATA_ROOT,
    DEVICE,
    DFL_REG_MAX,
    EARLY_STOP_PATIENCE,
    EMBED_DIM,
    EMBED_LOSS_WEIGHT,
    EPOCHS,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    GRAD_ACCUM_STEPS,
    IMG_SIZE,
    LEARNING_RATE,
    LLRD_GAMMA,
    NMS_THRESHOLD,
    REG_LOSS_WEIGHT,
    SAVE_PATH,
    VAL_INTERVAL,
    WEIGHT_DECAY,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_VALUE = 114
SEED = 42
VARIANT = "small"
TAL_TOPK = 10
TAL_ALPHA = 0.5
TAL_BETA = 6.0
MIN_POS_CELLS = 4
TINY_Q_TOL = 1.0
EVAL_IOU_THRS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
BUCKET_EDGES = ((16.0, math.inf), (32.0, 16.0), (96.0, 32.0), (math.inf, 96.0))
BUCKET_NAMES = ("tiny<16px", "small16-32px", "medium32-96px", "large>=96px")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class DetDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_file, img_size=960,
                 augment=True, patch_size=16, image_mean=None,
                 image_std=None, pad_value=114):
        if not isinstance(img_size, int) or img_size <= 0 or img_size % patch_size:
            raise ValueError("img_size must be a positive multiple of patch_size")
        if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(f"image and label directories must exist: {img_dir}, {label_dir}")

        self.img_dir = os.fspath(img_dir)
        self.label_dir = os.fspath(label_dir)
        self.img_size = img_size
        self.augment = bool(augment)
        self.patch_size = patch_size
        self.image_mean = tuple(IMAGENET_MEAN if image_mean is None else image_mean)
        self.image_std = tuple(IMAGENET_STD if image_std is None else image_std)
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must each contain three values")
        if not all(math.isfinite(v) for v in (*self.image_mean, *self.image_std)):
            raise ValueError("image_mean and image_std must be finite")
        if any(v <= 0 for v in self.image_std):
            raise ValueError("image_std values must be positive")
        if isinstance(pad_value, bool) or not isinstance(pad_value, (int, float)):
            raise ValueError("pad_value must be an integer in [0, 255]")
        if not math.isfinite(pad_value) or not 0 <= pad_value <= 255 or int(pad_value) != pad_value:
            raise ValueError("pad_value must be an integer in [0, 255]")
        self.pad_value = int(pad_value)

        with open(class_file, "r", encoding="utf-8") as f:
            self.categories = [line.strip() for line in f if line.strip()]
        if not self.categories:
            raise ValueError(f"class file is empty: {class_file}")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"duplicate class names in {class_file}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = sorted(
            name for name in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.img_dir, name))
            and os.path.splitext(name)[1].lower() in extensions
        )
        if not self.files:
            raise ValueError(f"no images found in {self.img_dir}")
        image_stems = {os.path.splitext(name)[0] for name in self.files}
        label_stems = {
            os.path.splitext(name)[0] for name in os.listdir(self.label_dir)
            if os.path.isfile(os.path.join(self.label_dir, name))
            and os.path.splitext(name)[1].lower() == ".txt"
        }
        missing = sorted(image_stems - label_stems)
        if missing:
            raise FileNotFoundError(f"missing label file for image {missing[0]}")
        orphaned = sorted(label_stems - image_stems)
        if orphaned:
            raise FileNotFoundError(f"label file has no image: {orphaned[0]}.txt")

        for name in self.files:
            image_path = os.path.join(self.img_dir, name)
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"invalid image {image_path}: {exc}") from exc
            label_path = self._label_path(name)
            _, labels = self._read_yolo(label_path)
            if any(label >= len(self.categories) for label in labels):
                raise ValueError(f"class id outside classes.txt in label for {name}")

    def __len__(self):
        return len(self.files)

    def _label_path(self, name):
        return os.path.join(self.label_dir, os.path.splitext(name)[0] + ".txt")

    @staticmethod
    def _read_yolo(path):
        boxes, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                parts = row.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("expected five fields")
                    class_id = int(parts[0])
                    cx, cy, width, height = map(float, parts[1:5])
                except ValueError as exc:
                    raise ValueError(f"invalid YOLO row in {path}: {row!r}") from exc
                if class_id < 0 or not all(math.isfinite(v) for v in (cx, cy, width, height)):
                    raise ValueError(f"invalid YOLO row in {path}: {row!r}")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"out-of-range YOLO row in {path}: {row!r}")
                x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(1.0, x2), min(1.0, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        return boxes, labels

    def _load_image(self, idx):
        name = self.files[idx]
        path = os.path.join(self.img_dir, name)
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.load()
        except Exception as exc:
            raise ValueError(f"invalid image {path}: {exc}") from exc
        boxes, labels = self._read_yolo(self._label_path(name))
        return image, boxes, labels

    def img_wh(self, idx):
        path = os.path.join(self.img_dir, self.files[idx])
        with Image.open(path) as image:
            return image.size

    def load_gt(self, idx):
        return self._read_yolo(self._label_path(self.files[idx]))

    def _augment(self, image, boxes, labels):
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))
        if random.random() < 0.1:
            image = ImageOps.grayscale(image).convert("RGB")
        if random.random() < 0.08:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
        return image, boxes, labels

    def __getitem__(self, idx):
        from runtime import prepare_image

        image, boxes, labels = self._load_image(idx)
        image, valid_mask, _ = prepare_image(
            image, self.img_size, self.patch_size, self.pad_value,
        )
        if self.augment:
            image, boxes, labels = self._augment(image, boxes, labels)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        pixel_values = torch.from_numpy(array.copy())
        mean = torch.tensor(self.image_mean).view(3, 1, 1)
        std = torch.tensor(self.image_std).view(3, 1, 1)
        pixel_values = (pixel_values - mean) / std

        target_boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target_labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": pixel_values, "target_boxes": target_boxes,
                "target_labels": target_labels, "valid_mask": valid_mask,
                "num_boxes": len(boxes), "src_idx": torch.tensor(idx)}


def collate_fn(batch):
    max_boxes = max(1, max(item.get("num_boxes", 0) for item in batch))
    target_boxes = torch.zeros(len(batch), max_boxes, 4)
    target_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    for index, item in enumerate(batch):
        count = item.get("num_boxes", 0)
        if count:
            target_boxes[index, :count] = item["target_boxes"]
            target_labels[index, :count] = item["target_labels"]
    output = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "target_boxes": target_boxes,
        "target_labels": target_labels,
        "valid_mask": torch.stack([item["valid_mask"] for item in batch]),
    }
    for key in ("src_idx",):
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch])
    return output


class ModelEMA:
    def __init__(self, model, decay=0.999, warmup=2000):
        self.decay = decay
        self.warmup = max(1, warmup)
        self.num_updates = 0
        self.ema = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.ema[name] = parameter.detach().float().clone()
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point():
                self.ema[name] = buffer.detach().float().clone()

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = self.decay * min(1.0, self.num_updates / self.warmup)
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if name in self.ema:
                self.ema[name].mul_(decay).add_(tensor.detach().float(), alpha=1 - decay)

    @torch.no_grad()
    def apply(self, model):
        backup = {}
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if name in self.ema:
                backup[name] = tensor.detach().clone()
                tensor.copy_(self.ema[name].to(tensor.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if name in backup:
                tensor.copy_(backup[name])


def discriminative_loss(embed_map, pos_mask, instance_ids, delta_v=0.5, delta_d=1.5):
    batch_size, _, height, width = embed_map.shape
    pull_loss = embed_map.new_zeros(())
    push_loss = embed_map.new_zeros(())
    n_instances = 0
    for batch_idx in range(batch_size):
        mask = pos_mask[batch_idx]
        if mask.sum() < 2:
            continue
        ids = instance_ids[batch_idx][mask]
        embeddings = embed_map[batch_idx].permute(1, 2, 0)[mask]
        means = []
        for instance_id in ids.unique():
            instance_embeddings = embeddings[ids == instance_id]
            if len(instance_embeddings) < 2:
                continue
            mean = instance_embeddings.mean(dim=0)
            means.append(mean)
            pull_loss = pull_loss + F.relu(
                torch.norm(instance_embeddings - mean.unsqueeze(0), dim=1) - delta_v
            ).mean()
            n_instances += 1
        if len(means) >= 2:
            distances = torch.cdist(torch.stack(means), torch.stack(means), p=2)
            off_diagonal = ~torch.eye(len(means), dtype=torch.bool, device=embed_map.device)
            push_loss = push_loss + F.relu(2 * delta_d - distances[off_diagonal]).mean()
    if n_instances:
        pull_loss = pull_loss / n_instances
        push_loss = push_loss / n_instances
    return pull_loss, push_loss, embed_map.abs().mean() * 0.001


def _compute_ciou(pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2):
    inter_w = (torch.minimum(pred_x2, gt_x2) - torch.maximum(pred_x1, gt_x1)).clamp(min=0)
    inter_h = (torch.minimum(pred_y2, gt_y2) - torch.maximum(pred_y1, gt_y1)).clamp(min=0)
    intersection = inter_w * inter_h
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = intersection / (pred_area + gt_area - intersection + 1e-7)
    enc_w = torch.maximum(pred_x2, gt_x2) - torch.minimum(pred_x1, gt_x1)
    enc_h = torch.maximum(pred_y2, gt_y2) - torch.minimum(pred_y1, gt_y1)
    center_distance = (((pred_x1 + pred_x2 - gt_x1 - gt_x2) * 0.5) ** 2
                       + ((pred_y1 + pred_y2 - gt_y1 - gt_y2) * 0.5) ** 2)
    pred_w, pred_h = pred_x2 - pred_x1, pred_y2 - pred_y1
    gt_w, gt_h = gt_x2 - gt_x1, gt_y2 - gt_y1
    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(gt_w / (gt_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7)), 2)
    alpha = v / (1 - iou + v + 1e-7)
    return iou - center_distance / (enc_w ** 2 + enc_h ** 2 + 1e-7) - alpha * v


def _dfl_loss(dist_logits, target, reg_max=DFL_REG_MAX):
    n_bins = 2 * reg_max + 1
    distribution = dist_logits.reshape(dist_logits.shape[0], 4, n_bins)
    clamped = target.clamp(0, 2 * reg_max)
    left = clamped.long().clamp(0, n_bins - 2)
    right = left + 1
    probabilities = F.softmax(distribution, dim=2)
    prob_left = probabilities.gather(2, left.unsqueeze(2)).squeeze(2)
    prob_right = probabilities.gather(2, right.unsqueeze(2)).squeeze(2)
    weight_right = (right.float() - clamped).clamp(0, 1)
    weight_left = (clamped - left.float()).clamp(0, 1)
    return -(weight_right * torch.log(prob_left + 1e-8)
             + weight_left * torch.log(prob_right + 1e-8)).mean()


def compute_class_weights(dataset, min_weight=0.7):
    if not 0 < min_weight <= 1:
        raise ValueError("min_weight must be in (0, 1]")
    num_classes = len(dataset.categories)
    counts = np.zeros(num_classes, dtype=np.float64)
    areas = np.zeros(num_classes, dtype=np.float64)
    for idx in range(len(dataset.files)):
        boxes, labels = dataset._read_yolo(dataset._label_path(dataset.files[idx]))
        for box, label in zip(boxes, labels):
            counts[label] += 1
            areas[label] += (box[2] - box[0]) * (box[3] - box[1])
    average_areas = np.where(counts > 0, areas / np.maximum(counts, 1), 0.25)
    raw = (1 / np.sqrt(counts + 1e-6)) / np.sqrt(np.sqrt(average_areas + 1e-6))
    weights = raw / raw.mean()
    raw_min = weights.min()
    if raw_min < min_weight:
        scale = (min_weight - 1) / (raw_min - 1)
        weights = 1 + (weights - 1) * scale
    return torch.tensor(weights, dtype=torch.float32)


def _compute_loss_scale(pred_boxes, pred_scores, pred_centerness, embed_map,
                        target_boxes, target_labels, device, valid_mask=None,
                        embed_weight=1.0, reg_weight=2.0, reg_dist=None,
                        class_weights=None, ignore_boxes=None):
    batch_size, _, height, width = pred_boxes.shape
    num_classes = pred_scores.shape[1]
    dtype = pred_boxes.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    center_x = grid_x.float() + 0.5
    center_y = grid_y.float() + 0.5
    pos_mask = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
    instance_ids = torch.full((batch_size, height, width), -1, dtype=torch.long, device=device)
    reg_target = torch.zeros(batch_size, 4, height, width, dtype=dtype, device=device)
    cls_target = torch.full((batch_size, height, width), -1, dtype=torch.long, device=device)
    iou_target = torch.zeros(batch_size, height, width, dtype=torch.float32, device=device)

    if valid_mask is not None:
        actual_h = valid_mask.any(dim=2).sum(dim=1).clamp(min=1).float()
        actual_w = valid_mask.any(dim=1).sum(dim=1).clamp(min=1).float()
    else:
        actual_h = torch.full((batch_size,), float(height), device=device)
        actual_w = torch.full((batch_size,), float(width), device=device)
    norm_x = center_x.unsqueeze(0) / actual_w.view(-1, 1, 1)
    norm_y = center_y.unsqueeze(0) / actual_h.view(-1, 1, 1)
    valid_gt = target_labels >= 0
    boxes = target_boxes
    gt_cx = ((boxes[..., 0] + boxes[..., 2]) * 0.5).unsqueeze(1)
    gt_cy = ((boxes[..., 1] + boxes[..., 3]) * 0.5).unsqueeze(1)
    grid_count = height * width
    max_boxes = target_boxes.shape[1]
    chunk = max(1, min(batch_size, int(6e6 // max(grid_count * max_boxes, 1))))

    for start in range(0, batch_size, chunk):
        end = min(batch_size, start + chunk)
        size = end - start
        valid_chunk = valid_gt[start:end].unsqueeze(1)
        box_chunk = boxes[start:end]
        gx = norm_x[start:end].reshape(size, grid_count, 1)
        gy = norm_y[start:end].reshape(size, grid_count, 1)
        in_box = ((gx >= box_chunk[..., 0].unsqueeze(1))
                  & (gx <= box_chunk[..., 2].unsqueeze(1))
                  & (gy >= box_chunk[..., 1].unsqueeze(1))
                  & (gy <= box_chunk[..., 3].unsqueeze(1)))
        if valid_mask is not None:
            in_box = in_box & valid_mask[start:end].reshape(size, grid_count, 1)
        in_box = in_box & valid_chunk

        decoded = pred_boxes[start:end].float()
        aw = actual_w[start:end].view(size, 1, 1)
        ah = actual_h[start:end].view(size, 1, 1)
        px1 = (norm_x[start:end] - decoded[:, 0] / aw).clamp(0, 1).reshape(size, grid_count)
        py1 = (norm_y[start:end] - decoded[:, 1] / ah).clamp(0, 1).reshape(size, grid_count)
        px2 = (norm_x[start:end] + decoded[:, 2] / aw).clamp(0, 1).reshape(size, grid_count)
        py2 = (norm_y[start:end] + decoded[:, 3] / ah).clamp(0, 1).reshape(size, grid_count)
        ix1 = torch.maximum(px1.unsqueeze(2), box_chunk[..., 0].unsqueeze(1))
        iy1 = torch.maximum(py1.unsqueeze(2), box_chunk[..., 1].unsqueeze(1))
        ix2 = torch.minimum(px2.unsqueeze(2), box_chunk[..., 2].unsqueeze(1))
        iy2 = torch.minimum(py2.unsqueeze(2), box_chunk[..., 3].unsqueeze(1))
        intersection = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        pred_area = ((px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)).unsqueeze(2)
        gt_area = ((box_chunk[..., 2] - box_chunk[..., 0]).clamp(min=0)
                   * (box_chunk[..., 3] - box_chunk[..., 1]).clamp(min=0)).unsqueeze(1)
        iou = intersection / (pred_area + gt_area - intersection + 1e-6)
        class_logits = pred_scores[start:end].float().permute(0, 2, 3, 1).reshape(size, grid_count, -1)
        gt_classes = target_labels[start:end].clamp(min=0).unsqueeze(1).expand(size, grid_count, max_boxes)
        class_scores = torch.gather(class_logits, 2, gt_classes).sigmoid()
        alignment = (class_scores.clamp(min=1e-8) ** TAL_ALPHA) * (iou.clamp(min=0) ** TAL_BETA)
        alignment = alignment.masked_fill(~in_box, 0)

        topk = min(TAL_TOPK, grid_count)
        top_indices = torch.topk(alignment, k=topk, dim=1).indices
        topk_mask = torch.zeros_like(alignment, dtype=torch.bool)
        topk_mask.scatter_(1, top_indices, True)
        topk_mask = topk_mask & (alignment > 0)
        max_alignment, assigned_gt = alignment.masked_fill(~topk_mask, 0).max(dim=2)
        assigned = max_alignment > 0
        one_hot = F.one_hot(assigned_gt.clamp(min=0), max_boxes) & assigned.unsqueeze(2)
        hits = one_hot.sum(dim=1)
        missing = (hits == 0) & valid_gt[start:end]
        few = (hits < MIN_POS_CELLS) & valid_gt[start:end]
        distance = (gx - gt_cx[start:end]) ** 2 + (gy - gt_cy[start:end]) ** 2
        if valid_mask is not None:
            distance = distance.masked_fill(~valid_mask[start:end].reshape(size, grid_count, 1)
                                            | ~valid_chunk, float("inf"))
        else:
            distance = distance.masked_fill(~valid_chunk, float("inf"))
        nearest = distance.argmin(dim=2)
        nearest_cell = distance.argmin(dim=1)
        row_offset = (torch.arange(size, device=device) * grid_count).unsqueeze(1)
        fallback = torch.zeros(size * grid_count, device=device)
        fallback.index_add_(0, (nearest_cell + row_offset).reshape(-1),
                            missing.reshape(-1).to(fallback.dtype))
        fcx = gt_cx[start:end].reshape(size, max_boxes) * actual_w[start:end].unsqueeze(1) - 0.5
        fcy = gt_cy[start:end].reshape(size, max_boxes) * actual_h[start:end].unsqueeze(1) - 0.5
        x0 = fcx.floor().clamp(0, width - 1).long()
        y0 = fcy.floor().clamp(0, height - 1).long()
        valid_flat = valid_mask[start:end].reshape(size, grid_count) if valid_mask is not None else None
        for dy in (0, 1):
            for dx in (0, 1):
                cell = ((y0 + dy).clamp(max=height - 1) * width
                        + (x0 + dx).clamp(max=width - 1))
                weight = few.to(fallback.dtype) * in_box.gather(1, cell.unsqueeze(1)).squeeze(1)
                if valid_flat is not None:
                    weight = weight * valid_flat.gather(1, cell)
                fallback.index_add_(0, (cell + row_offset).reshape(-1), weight.reshape(-1))
        added = fallback.view(size, grid_count) > 0
        assigned_gt = torch.where(assigned, assigned_gt, nearest)
        assigned = assigned | added
        assigned_iou = iou.gather(2, assigned_gt.clamp(min=0).unsqueeze(2)).squeeze(2)
        iou_target[start:end] = (assigned_iou * assigned).view(size, height, width)

        assigned_view = assigned.view(size, height, width)
        assigned_indices = assigned_gt.view(size, height, width)
        assigned_classes = target_labels[start:end].gather(1, assigned_gt.clamp(min=0)).view(size, height, width)
        cls_target[start:end] = torch.where(assigned_view, assigned_classes, -1)
        instance_ids[start:end] = torch.where(assigned_view, assigned_indices, -1)
        pos_mask[start:end] = assigned_view
        positive = assigned.to(dtype).view(size, height, width)
        gathered = assigned_gt.clamp(min=0)
        bx1 = box_chunk[..., 0].gather(1, gathered).view(size, height, width)
        by1 = box_chunk[..., 1].gather(1, gathered).view(size, height, width)
        bx2 = box_chunk[..., 2].gather(1, gathered).view(size, height, width)
        by2 = box_chunk[..., 3].gather(1, gathered).view(size, height, width)
        reg_target[start:end, 0] = (norm_x[start:end] - bx1) * positive
        reg_target[start:end, 1] = (norm_y[start:end] - by1) * positive
        reg_target[start:end, 2] = (bx2 - norm_x[start:end]) * positive
        reg_target[start:end, 3] = (by2 - norm_y[start:end]) * positive

    class_target = torch.zeros(batch_size, height, width, num_classes, dtype=torch.float32, device=device)
    positive_classes = cls_target >= 0
    if positive_classes.any():
        class_target[positive_classes] = F.one_hot(cls_target[positive_classes], num_classes).float()
        if CLS_IOU_AWARE:
            quality = iou_target.detach().pow(CLS_IOU_POWER).clamp(min=CLS_IOU_FLOOR, max=1)
            class_target = class_target * quality.unsqueeze(-1)
    scores = pred_scores.permute(0, 2, 3, 1)
    cross_entropy = F.binary_cross_entropy_with_logits(scores, class_target, reduction="none")
    probability = torch.sigmoid(scores)
    probability_target = probability * class_target + (1 - probability) * (1 - class_target)
    focal_weight = 0.25 * (1 - probability_target) ** 2
    if ignore_boxes is not None:
        ignored = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
        for box_idx in range(ignore_boxes.shape[1]):
            box = ignore_boxes[:, box_idx]
            ignored |= ((norm_x >= box[:, 0].view(-1, 1, 1))
                        & (norm_x <= box[:, 2].view(-1, 1, 1))
                        & (norm_y >= box[:, 1].view(-1, 1, 1))
                        & (norm_y <= box[:, 3].view(-1, 1, 1)))
        focal_weight = focal_weight * (~(ignored & (cls_target < 0))).unsqueeze(-1)
    num_positive = int(positive_classes.sum().item())
    normalizer = max(num_positive, 1)
    if class_weights is not None:
        if class_weights.numel() != num_classes:
            raise ValueError("class_weights size does not match the number of classes")
        weight_map = torch.ones_like(class_target)
        if num_positive:
            sample_weights = class_weights[cls_target[positive_classes]].unsqueeze(1)
            weight_map[positive_classes] = (sample_weights * class_target[positive_classes]
                                            + 1 - class_target[positive_classes])
        cls_loss = (focal_weight * cross_entropy * weight_map).sum() / normalizer
    else:
        cls_loss = (focal_weight * cross_entropy).sum() / normalizer

    if num_positive:
        if reg_dist is None:
            raise ValueError("reg_dist is required for DFL training")
        pos_batch, pos_y, pos_x = torch.where(pos_mask)
        aw, ah = actual_w[pos_batch], actual_h[pos_batch]
        scale = torch.stack((aw, ah, aw, ah), dim=1)
        positive_flat = pos_mask.reshape(batch_size, -1)
        pred_flat = pred_boxes.reshape(batch_size, 4, -1).transpose(1, 2)[positive_flat]
        target_flat = reg_target.reshape(batch_size, 4, -1).transpose(1, 2)[positive_flat]
        target_cells = (target_flat * scale).clamp(0, 2 * DFL_REG_MAX)
        smooth_l1 = F.smooth_l1_loss(pred_flat, target_cells, beta=1.0)
        dist_flat = reg_dist.reshape(batch_size, -1, height, width).permute(0, 2, 3, 1)
        dist_flat = dist_flat.reshape(batch_size * height * width, -1)
        dfl = _dfl_loss(dist_flat[pos_mask.reshape(-1)], target_cells, DFL_REG_MAX)
        pred_norm = pred_flat / scale
        target_norm = target_flat
        grid_norm_x = (pos_x.float() + 0.5) / aw
        grid_norm_y = (pos_y.float() + 0.5) / ah
        pred_x1 = (grid_norm_x - pred_norm[:, 0]).clamp(0, 1)
        pred_y1 = (grid_norm_y - pred_norm[:, 1]).clamp(0, 1)
        pred_x2 = (grid_norm_x + pred_norm[:, 2]).clamp(0, 1)
        pred_y2 = (grid_norm_y + pred_norm[:, 3]).clamp(0, 1)
        gt_x1 = (grid_norm_x - target_norm[:, 0]).clamp(0, 1)
        gt_y1 = (grid_norm_y - target_norm[:, 1]).clamp(0, 1)
        gt_x2 = (grid_norm_x + target_norm[:, 2]).clamp(0, 1)
        gt_y2 = (grid_norm_y + target_norm[:, 3]).clamp(0, 1)
        ciou = _compute_ciou(pred_x1, pred_y1, pred_x2, pred_y2, gt_x1, gt_y1, gt_x2, gt_y2)
        reg_loss = 1 - ciou.mean() + 0.5 * smooth_l1 + 0.25 * dfl

        short_side_pixels = torch.minimum(target_norm[:, 0] + target_norm[:, 2],
                                          target_norm[:, 1] + target_norm[:, 3]) * IMG_SIZE
        tolerance = torch.clamp(short_side_pixels * 0.15, max=TINY_Q_TOL) / IMG_SIZE
        quality_x1 = torch.minimum(gt_x1, torch.maximum(pred_x1, gt_x1 - tolerance))
        quality_y1 = torch.minimum(gt_y1, torch.maximum(pred_y1, gt_y1 - tolerance))
        quality_x2 = torch.maximum(gt_x2, torch.minimum(pred_x2, gt_x2 + tolerance))
        quality_y2 = torch.maximum(gt_y2, torch.minimum(pred_y2, gt_y2 + tolerance))
        iou_quality = torch.maximum(
            _compute_ciou(pred_x1, pred_y1, pred_x2, pred_y2,
                          quality_x1, quality_y1, quality_x2, quality_y2), ciou
        ).detach().clamp(0, 1)
        finite = iou_quality.isfinite()
        centerness = pred_centerness.squeeze(1)[pos_mask]
        cent_loss = F.binary_cross_entropy(centerness[finite], iou_quality[finite])
    else:
        reg_loss = pred_boxes.new_zeros(())
        cent_loss = pred_boxes.new_zeros(())

    pull_loss, push_loss, embed_reg_loss = discriminative_loss(embed_map, pos_mask, instance_ids)
    total = (cls_loss + reg_weight * reg_loss + cent_loss
             + embed_weight * (pull_loss + push_loss + embed_reg_loss))
    return (total, cls_loss, reg_loss, cent_loss, pull_loss, push_loss,
            embed_reg_loss, num_positive)


def _upsample_valid_mask(valid_mask, up):
    if up == 1:
        return valid_mask
    return F.interpolate(valid_mask.float().unsqueeze(1), scale_factor=up,
                         mode="nearest").squeeze(1).bool()


def compute_loss(pred_boxes, pred_scores, pred_centerness, embed_map,
                 target_boxes, target_labels, device, valid_mask=None,
                 embed_weight=1.0, reg_weight=2.0, class_weights=None, reg_dist=None):
    for name, value in (("pred_boxes", pred_boxes), ("pred_scores", pred_scores),
                        ("pred_centerness", pred_centerness), ("embed_map", embed_map)):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must contain exactly two scales")
    if reg_dist is not None and (not isinstance(reg_dist, (list, tuple)) or len(reg_dist) != 2):
        raise ValueError("reg_dist must contain exactly two scales")
    coarse_height, coarse_width = pred_boxes[0].shape[2:]
    if valid_mask is None:
        actual_w = torch.full((target_boxes.shape[0],), coarse_width,
                              dtype=target_boxes.dtype, device=target_boxes.device)
        actual_h = torch.full((target_boxes.shape[0],), coarse_height,
                              dtype=target_boxes.dtype, device=target_boxes.device)
    else:
        actual_w = valid_mask.any(dim=1).sum(dim=1).clamp(min=1).to(target_boxes.dtype)
        actual_h = valid_mask.any(dim=2).sum(dim=1).clamp(min=1).to(target_boxes.dtype)
    box_side_cells = torch.maximum(
        (target_boxes[..., 2] - target_boxes[..., 0]) * actual_w.unsqueeze(1),
        (target_boxes[..., 3] - target_boxes[..., 1]) * actual_h.unsqueeze(1),
    )
    class_numerator = None
    accumulated = None
    total_positive = 0
    for scale_idx in range(2):
        keep_fine = box_side_cells < 2.0
        dropped = (keep_fine if scale_idx == 0 else ~keep_fine) & (target_labels >= 0)
        routed_labels = target_labels.masked_fill(dropped, -1)
        ignored_boxes = torch.where(
            dropped.unsqueeze(-1), target_boxes, torch.full_like(target_boxes, 2.0),
        )
        scale_mask = (_upsample_valid_mask(valid_mask, pred_boxes[scale_idx].shape[2] // coarse_height)
                      if valid_mask is not None else None)
        result = _compute_loss_scale(
            pred_boxes[scale_idx], pred_scores[scale_idx], pred_centerness[scale_idx],
            embed_map[scale_idx], target_boxes, routed_labels, device,
            valid_mask=scale_mask, embed_weight=embed_weight, reg_weight=reg_weight,
            reg_dist=None if reg_dist is None else reg_dist[scale_idx],
            class_weights=class_weights, ignore_boxes=ignored_boxes)
        _, cls_loss, reg_loss, cent_loss, pull_loss, push_loss, embed_reg_loss, positive = result
        if class_numerator is None:
            class_numerator = torch.zeros_like(cls_loss)
            accumulated = [torch.zeros_like(value) for value in
                           (reg_loss, cent_loss, pull_loss, push_loss, embed_reg_loss)]
        weight = float(positive)
        class_numerator = class_numerator + cls_loss * weight
        total_positive += positive
        for index, value in enumerate((reg_loss, cent_loss, pull_loss, push_loss, embed_reg_loss)):
            accumulated[index] = accumulated[index] + value * weight
    normalizer = max(total_positive, 1)
    cls_loss = class_numerator / normalizer
    reg_loss, cent_loss, pull_loss, push_loss, embed_reg_loss = [value / normalizer for value in accumulated]
    total = (cls_loss + reg_weight * reg_loss + cent_loss
             + embed_weight * (pull_loss + push_loss + embed_reg_loss))
    return total, cls_loss, reg_loss, cent_loss, pull_loss, push_loss, embed_reg_loss


def _longest_side_pixels(boxes_xyxy):
    return torch.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                         boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).cpu().numpy()


def _ap_from_pr(precision, recall):
    mean_recall = np.concatenate(([0.0], recall, [1.0]))
    mean_precision = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(len(mean_precision) - 1, 0, -1):
        mean_precision[idx - 1] = max(mean_precision[idx - 1], mean_precision[idx])
    indices = np.where(mean_recall[1:] != mean_recall[:-1])[0]
    return float(np.sum((mean_recall[indices + 1] - mean_recall[indices]) * mean_precision[indices + 1]))


def _iou_batch(a, b):
    x1 = torch.maximum(a[:, 0:1], b[:, 0].unsqueeze(0))
    y1 = torch.maximum(a[:, 1:2], b[:, 1].unsqueeze(0))
    x2 = torch.minimum(a[:, 2:3], b[:, 2].unsqueeze(0))
    y2 = torch.minimum(a[:, 3:4], b[:, 3].unsqueeze(0))
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / (area_a.unsqueeze(1) + area_b.unsqueeze(0) - intersection + 1e-6)


def _nms(boxes, scores, thr):
    boxes = torch.as_tensor(boxes)
    scores = torch.as_tensor(scores)
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return _torchvision_nms(boxes, scores, thr)


@torch.no_grad()
def evaluate(model, dataset, device, eval_batch=8, eval_workers=0):
    from runtime import decode_predictions

    if eval_batch < 1 or eval_workers < 0:
        raise ValueError("eval_batch must be positive and eval_workers non-negative")
    device = torch.device(device)
    cuda = device.type == "cuda"
    amp_enabled = cuda and torch.cuda.is_available()
    previous_training = model.training
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    loader_kwargs = {"batch_size": eval_batch, "collate_fn": collate_fn,
                     "num_workers": eval_workers}
    if eval_workers:
        loader_kwargs.update(pin_memory=cuda, persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)
    num_classes = len(dataset.categories)
    class_data = {(class_id, threshold): {"scores": [], "matches": [], "n_gt": 0}
                  for class_id in range(num_classes) for threshold in EVAL_IOU_THRS}
    bucket_data = {(name, threshold): {"n_gt": 0, "n_tp": 0}
                   for name in BUCKET_NAMES for threshold in EVAL_IOU_THRS}
    forward_time = 0.0
    start_time = time.perf_counter()

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    def finalize_image(source_idx, detections):
        width, height = dataset.img_wh(source_idx)
        gt_normalized, gt_labels = dataset.load_gt(source_idx)
        gt_boxes = torch.tensor(gt_normalized, dtype=torch.float32).reshape(-1, 4)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        gt_pixels = gt_boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        sides = _longest_side_pixels(gt_pixels)
        buckets = np.empty(len(sides), dtype=object)
        for idx, side in enumerate(sides):
            for name, (upper, lower) in zip(BUCKET_NAMES, BUCKET_EDGES):
                if lower <= side < upper:
                    buckets[idx] = name
                    break
        for class_id in range(num_classes):
            class_gt = gt_labels == class_id
            n_gt = int(class_gt.sum())
            pixels = gt_pixels[class_gt]
            class_mask = class_gt.numpy()
            class_sides = sides[class_mask]
            class_buckets = buckets[class_mask]
            for threshold in EVAL_IOU_THRS:
                class_data[(class_id, threshold)]["n_gt"] += n_gt
            for name, (upper, lower) in zip(BUCKET_NAMES, BUCKET_EDGES):
                count = int(((class_sides >= lower) & (class_sides < upper)).sum())
                for threshold in EVAL_IOU_THRS:
                    bucket_data[(name, threshold)]["n_gt"] += count
            candidates = detections[class_id]
            if candidates:
                det_boxes = torch.stack([item[0] for item in candidates])
                det_scores = torch.stack([item[1] for item in candidates])
                pixel_scale = torch.tensor(
                    [width, height, width, height], dtype=det_boxes.dtype,
                )
                keep = _nms(det_boxes * pixel_scale, det_scores, NMS_THRESHOLD)[:500]
                det_boxes, det_scores = det_boxes[keep], det_scores[keep]
            else:
                det_boxes = det_scores = None
            if det_boxes is None:
                continue
            scores = det_scores.numpy()
            order = np.argsort(-scores)
            if n_gt:
                det_pixels = det_boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
                overlaps = _iou_batch(det_pixels, pixels).numpy()
            for threshold in EVAL_IOU_THRS:
                matches = np.zeros(len(det_boxes), dtype=np.int64)
                matched_buckets = np.empty(len(det_boxes), dtype=object)
                matched_buckets[:] = None
                if n_gt:
                    gt_matched = np.zeros(n_gt, dtype=bool)
                    for det_idx in order:
                        row = overlaps[det_idx].copy()
                        row[gt_matched] = -1
                        gt_idx = int(row.argmax())
                        if row[gt_idx] >= threshold:
                            matches[det_idx] = 1
                            gt_matched[gt_idx] = True
                            matched_buckets[det_idx] = class_buckets[gt_idx]
                data = class_data[(class_id, threshold)]
                data["scores"].extend(scores.tolist())
                data["matches"].extend(matches.tolist())
                for name in BUCKET_NAMES:
                    bucket_data[(name, threshold)]["n_tp"] += int(
                        np.sum((matches == 1) & (matched_buckets == name)))

    try:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device)
            synchronize()
            tick = time.perf_counter()
            autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if amp_enabled else nullcontext())
            with autocast:
                reg_pred, _, cls_pred, centerness_pred, _, _ = model(pixel_values)
            synchronize()
            forward_time += time.perf_counter() - tick
            if not isinstance(reg_pred, (list, tuple)) or len(reg_pred) != 2:
                raise ValueError("evaluate requires the fixed two-scale P2 model")
            reg_pred = [value.float() for value in reg_pred]
            cls_pred = [value.float() for value in cls_pred]
            centerness_pred = [value.float() for value in centerness_pred]
            for batch_idx in range(pixel_values.shape[0]):
                source_idx = int(batch["src_idx"][batch_idx])
                original_size = dataset.img_wh(source_idx)
                detections = decode_predictions(
                    [value[batch_idx] for value in reg_pred],
                    [value[batch_idx] for value in cls_pred],
                    [value[batch_idx] for value in centerness_pred],
                    valid_mask[batch_idx], original_size, num_classes,
                    conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD,
                )
                finalize_image(source_idx, detections)
    finally:
        model.train(previous_training)
        torch.set_num_threads(previous_threads)

    results, class_details = {}, {}
    for threshold in EVAL_IOU_THRS:
        aps, recalls, details = [], [], []
        for class_id in range(num_classes):
            data = class_data[(class_id, threshold)]
            n_gt = data["n_gt"]
            matches = np.asarray(data["matches"], dtype=np.int64)
            n_tp = int(matches.sum())
            recall = n_tp / n_gt if n_gt else 0.0
            ap = 0.0
            if n_gt and len(matches):
                order = np.argsort(np.asarray(data["scores"]))[::-1]
                true_positive = np.cumsum(matches[order])
                false_positive = np.cumsum(1 - matches[order])
                precision = true_positive / (true_positive + false_positive + 1e-6)
                ap = _ap_from_pr(precision, true_positive / n_gt)
            details.append({"cls_id": class_id, "cls_name": dataset.categories[class_id],
                            "ap": ap, "recall": recall})
            if n_gt:
                aps.append(ap)
                recalls.append(recall)
        results[f"mAP@{threshold:g}"] = float(np.mean(aps)) if aps else 0.0
        results[f"recall@{threshold:g}"] = float(np.mean(recalls)) if recalls else 0.0
        class_details[threshold] = details
    results["mAP@0.5:0.95"] = float(np.mean([results[f"mAP@{value:g}"] for value in EVAL_IOU_THRS]))
    results["recall@0.5:0.95"] = float(np.mean([results[f"recall@{value:g}"] for value in EVAL_IOU_THRS]))
    for name in BUCKET_NAMES:
        results[f"recall@0.5[{name}]"] = (
            bucket_data[(name, 0.5)]["n_tp"] / bucket_data[(name, 0.5)]["n_gt"]
            if bucket_data[(name, 0.5)]["n_gt"] else 0.0)
        recalls = [bucket_data[(name, value)]["n_tp"] / bucket_data[(name, value)]["n_gt"]
                   if bucket_data[(name, value)]["n_gt"] else 0.0 for value in EVAL_IOU_THRS]
        results[f"recall@0.5:0.95[{name}]"] = float(np.mean(recalls))
    elapsed = time.perf_counter() - start_time
    timing = {"n_images": len(dataset.files), "eval_time_s": round(elapsed, 3),
              "images_per_s": round(len(dataset.files) / max(elapsed, 1e-9), 3),
              "forward_time_s": round(forward_time, 3),
              "forward_images_per_s": round(len(dataset.files) / max(forward_time, 1e-9), 3)}
    return results, class_details, timing


def _load_model_weights(model, ckpt, expected_categories=None):
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dictionary")
    categories = ckpt.get("categories")
    if not isinstance(categories, (list, tuple)) or not categories:
        raise ValueError("checkpoint must contain categories")
    if expected_categories is not None and list(categories) != list(expected_categories):
        raise ValueError("checkpoint categories do not match classes.txt")
    architecture = ckpt.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint must contain architecture metadata")
    expected_architecture = {
        "variant": getattr(model, "variant", None),
        "use_p2": bool(getattr(model, "use_p2", False)),
        "reg_max": model.det_head.reg_max,
        "num_classes": model.num_classes,
        "embed_dim": model.det_head.embed_dim,
        "img_size": model.img_size,
        "num_inter_layers": getattr(model, "num_inter_layers", None),
    }
    for key, actual in expected_architecture.items():
        if key not in architecture:
            raise ValueError(f"checkpoint architecture is missing {key}")
        if architecture[key] != actual:
            raise ValueError(f"checkpoint architecture mismatch for {key}: {architecture[key]!r} != {actual!r}")
    if not expected_architecture["use_p2"]:
        raise ValueError("the released detector requires P2")
    required = ("fpn", "det_head", "p2_conv", "p2_head", "p2_stem", "backbone")
    for key in required:
        if key not in ckpt:
            raise ValueError(f"checkpoint is missing {key}")
    if not isinstance(ckpt["backbone"], dict) or not ckpt["backbone"]:
        raise ValueError("checkpoint must contain a complete backbone state dict")
    model.fpn.load_state_dict(ckpt["fpn"], strict=True)
    model.det_head.load_state_dict(ckpt["det_head"], strict=True)
    model.p2_conv.load_state_dict(ckpt["p2_conv"], strict=True)
    model.p2_head.load_state_dict(ckpt["p2_head"], strict=True)
    model.p2_stem.load_state_dict(ckpt["p2_stem"], strict=True)
    model.backbone.load_state_dict(ckpt["backbone"], strict=True)


def training_paths(data_root):
    root = os.path.abspath(os.fspath(data_root))
    return {
        "train_images": os.path.join(root, "images", "train"),
        "train_labels": os.path.join(root, "labels", "train"),
        "val_images": os.path.join(root, "images", "val"),
        "val_labels": os.path.join(root, "labels", "val"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the fixed two-stage object detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--classes", default=None)
    parser.add_argument("--output-dir", default=os.path.dirname(SAVE_PATH))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--unfreeze-blocks", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--stage", choices=("frozen", "finetune"), default="frozen")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.grad_accum < 1:
        parser.error("batch-size and grad-accum must be positive")
    if args.workers < 0 or args.eval_workers < 0:
        parser.error("workers and eval-workers must be non-negative")
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    if args.unfreeze_blocks is not None and not 0 <= args.unfreeze_blocks <= 12:
        parser.error("unfreeze-blocks must be in [0, 12]")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.backbone_lr) or args.backbone_lr <= 0:
        parser.error("learning rates must be finite and positive")
    if args.stage == "finetune" and not args.resume:
        parser.error("--resume is required for --stage finetune")
    if args.stage == "finetune":
        args.epochs = FINETUNE_EPOCHS
        args.unfreeze_blocks = 12
        args.lr = FINETUNE_LR
    else:
        args.epochs = EPOCHS if args.epochs is None else args.epochs
        args.unfreeze_blocks = 0
    args.data_root = os.path.abspath(args.data_root)
    args.classes = os.path.abspath(args.classes or os.path.join(args.data_root, "classes.txt"))
    args.output_dir = os.path.abspath(args.output_dir)
    return args


def _autocast(device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main():
    args = parse_args()
    set_seed(args.seed)
    device_arg = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(device_arg)
    paths = training_paths(args.data_root)
    for path in (*paths.values(), args.classes):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    print(f"[train] data_root={args.data_root}", flush=True)
    from runtime import load_categories
    categories = load_categories(args.classes)

    checkpoint = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(args.resume)
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("backbone"), dict):
            raise ValueError("resume checkpoint must contain a complete backbone")
        if not checkpoint["backbone"]:
            raise ValueError("resume checkpoint must contain a complete backbone")
    from model import LbotDetector
    model = LbotDetector(
        variant=VARIANT, num_classes=len(categories), embed_dim=EMBED_DIM,
        unfreeze_blocks=args.unfreeze_blocks, img_size=IMG_SIZE,
        device=device, dtype=torch.float32,
        backbone_path=None if checkpoint is not None else os.environ.get("LBOT_BACKBONE_PATH"),
        backbone_state_dict=checkpoint["backbone"] if checkpoint is not None else None,
    ).to(device)
    if checkpoint is not None:
        _load_model_weights(model, checkpoint, expected_categories=categories)

    train_dataset = DetDataset(
        paths["train_images"], paths["train_labels"], args.classes,
        img_size=IMG_SIZE, augment=True, patch_size=model.patch_size)
    val_dataset = DetDataset(
        paths["val_images"], paths["val_labels"], args.classes,
        img_size=IMG_SIZE, augment=False, patch_size=model.patch_size)
    print(
        f"[train] train_images={len(train_dataset)} val_images={len(val_dataset)}",
        flush=True,
    )
    loader_kwargs = {"batch_size": args.batch_size, "shuffle": True,
                     "num_workers": args.workers, "collate_fn": collate_fn}
    if args.workers:
        loader_kwargs.update(pin_memory=device.type == "cuda", persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(train_dataset, **loader_kwargs)
    class_weights = compute_class_weights(train_dataset).to(device)

    detector_modules = [model.fpn, model.det_head, model.p2_conv, model.p2_stem, model.p2_head]
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}

    def split_decay(parameters):
        decay, no_decay = [], []
        for parameter in parameters:
            if parameter.requires_grad:
                (decay if parameter.ndim > 1 else no_decay).append(parameter)
        return decay, no_decay

    head_lr = FINETUNE_LR if args.stage == "finetune" else args.lr
    backbone_lr = (args.backbone_lr * FINETUNE_LR / LEARNING_RATE
                   if args.stage == "finetune" else args.backbone_lr)
    groups = []
    decay, no_decay = split_decay([parameter for module in detector_modules for parameter in module.parameters()])
    if decay:
        groups.append({"params": decay, "lr": head_lr, "weight_decay": WEIGHT_DECAY})
    if no_decay:
        groups.append({"params": no_decay, "lr": head_lr, "weight_decay": 0.0})
    blocks = list(model.backbone.blocks)
    assigned = set()
    for index, block in enumerate(blocks):
        scale = LLRD_GAMMA ** (len(blocks) - 1 - index)
        parameters = [parameter for parameter in block.parameters() if parameter.requires_grad]
        assigned.update(id(parameter) for parameter in parameters)
        decay, no_decay = split_decay(parameters)
        if decay:
            groups.append({"params": decay, "lr": backbone_lr * scale, "weight_decay": WEIGHT_DECAY})
        if no_decay:
            groups.append({"params": no_decay, "lr": backbone_lr * scale, "weight_decay": 0.0})
    other = [parameter for parameter in model.backbone.parameters()
             if parameter.requires_grad and id(parameter) not in assigned]
    decay, no_decay = split_decay(other)
    minimum_scale = LLRD_GAMMA ** len(blocks)
    if decay:
        groups.append({"params": decay, "lr": backbone_lr * minimum_scale, "weight_decay": WEIGHT_DECAY})
    if no_decay:
        groups.append({"params": no_decay, "lr": backbone_lr * minimum_scale, "weight_decay": 0.0})
    for group in groups:
        group["param_names"] = [parameter_names[id(parameter)] for parameter in group["params"]]
    optimizer = torch.optim.AdamW(groups)
    if args.stage == "finetune":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)
        start_epoch = 0
    else:
        warmup_epochs = min(2, args.epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
             torch.optim.lr_scheduler.CosineAnnealingLR(
                 optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6)],
            milestones=[warmup_epochs])
        start_epoch = int(checkpoint.get("epoch", 0)) if checkpoint is not None else 0
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError("checkpoint epoch is outside the frozen training schedule")
        for _ in range(start_epoch):
            scheduler.step()

    best_metric = float(checkpoint.get("best_metric", 0.0)) if checkpoint is not None else None
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best.pth")
    last_path = os.path.join(args.output_dir, "last.pth")

    def state_dict(module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    def build_checkpoint(epoch, metric):
        return {
            "fpn": state_dict(model.fpn),
            "det_head": state_dict(model.det_head),
            "p2_conv": state_dict(model.p2_conv),
            "p2_head": state_dict(model.p2_head),
            "p2_stem": state_dict(model.p2_stem),
            "backbone": state_dict(model.backbone),
            "categories": list(categories),
            "architecture": {
                "variant": VARIANT, "use_p2": True, "reg_max": DFL_REG_MAX,
                "num_classes": len(categories), "embed_dim": EMBED_DIM,
                "img_size": IMG_SIZE, "num_inter_layers": model.num_inter_layers,
            },
            "preprocessing": {
                "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                "pad": PAD_VALUE, "image_size": IMG_SIZE,
            },
            "epoch": epoch,
            "best_metric": float(metric),
        }

    if checkpoint is not None and os.path.abspath(args.resume) != best_path:
        torch.save(build_checkpoint(start_epoch, best_metric or 0.0), best_path)
    ema = ModelEMA(model)
    patience = 0
    model.train()
    for epoch in range(start_epoch, args.epochs):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0
        steps_since_step = 0
        for batch_idx, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with _autocast(device):
                reg_pred, reg_dist, cls_pred, centerness_pred, _, embed_map = model(pixel_values)
            reg_pred = [value.float() for value in reg_pred]
            reg_dist = [value.float() for value in reg_dist]
            cls_pred = [value.float() for value in cls_pred]
            centerness_pred = [value.float() for value in centerness_pred]
            embed_map = [value.float() for value in embed_map]
            loss, cls_loss, reg_loss, cent_loss, _, _, _ = compute_loss(
                reg_pred, cls_pred, centerness_pred, embed_map,
                batch["target_boxes"].to(device), batch["target_labels"].to(device), device,
                valid_mask=batch["valid_mask"].to(device), embed_weight=EMBED_LOSS_WEIGHT,
                reg_weight=REG_LOSS_WEIGHT, class_weights=class_weights, reg_dist=reg_dist)
            cycle_start = (batch_idx // args.grad_accum) * args.grad_accum
            accumulation_size = min(args.grad_accum, len(loader) - cycle_start)
            (loss / accumulation_size).backward()
            epoch_loss += loss.detach().item()
            steps_since_step += 1
            if steps_since_step == args.grad_accum or batch_idx + 1 == len(loader):
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                steps_since_step = 0
            progress.set_postfix(loss=f"{loss.detach().item():.4f}",
                                 cls=f"{cls_loss.detach().item():.3f}",
                                 reg=f"{reg_loss.detach().item():.3f}",
                                 cent=f"{cent_loss.detach().item():.3f}")
        scheduler.step()
        print(f"Epoch {epoch + 1} | loss={epoch_loss / len(loader):.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.6g} | time={time.perf_counter() - tick:.1f}s")

        if (epoch + 1) % VAL_INTERVAL:
            continue
        backup = ema.apply(model)
        try:
            results, class_details, timing = evaluate(
                model, val_dataset, device,
                eval_batch=args.batch_size, eval_workers=args.eval_workers)
            metric = results["mAP@0.5"]
            improved = best_metric is None or metric > best_metric
            print(f"Validation | mAP@0.5={metric:.4f} | recall@0.5={results['recall@0.5']:.4f} | "
                  f"mAP@0.5:0.95={results['mAP@0.5:0.95']:.4f} | "
                  f"time={timing['eval_time_s']:.1f}s")
            for detail in class_details[0.5]:
                print(f"  {detail['cls_name']}: AP={detail['ap']:.4f}, recall={detail['recall']:.4f}")
            for name in BUCKET_NAMES:
                print(f"  {name}: recall={results[f'recall@0.5[{name}]']:.4f}")
            if improved:
                best_metric = metric
                patience = 0
                torch.save(build_checkpoint(epoch + 1, best_metric), best_path)
                print(f"Saved best checkpoint: {best_path}")
            else:
                patience += 1
        finally:
            ema.restore(model, backup)
            model.train()
        if not improved and patience >= EARLY_STOP_PATIENCE:
            torch.save(build_checkpoint(epoch + 1, best_metric), last_path)
            print(f"Early stopping after {patience} validations without improvement: {last_path}")
            break
    print(f"Training complete | best mAP@0.5={best_metric if best_metric is not None else 0.0:.4f}")


if __name__ == "__main__":
    main()
