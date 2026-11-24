import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.lingbot_vision.lingbot_vision import load_pretrained_backbone
from third_party.lingbot_vision.lingbot_vision.loader import (
    PRETRAINED_VARIANTS, _normalize_variant, _resolve_dtype, load_config,
)
from third_party.lingbot_vision.lingbot_vision.build import build_backbone_from_cfg


class DetHead(nn.Module):
    def __init__(self, in_channels=256, feat_channels=256,
                 num_classes=1, embed_dim=8):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        self.reg_max = 8
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.vis_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.bbox_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4 * (2 * self.reg_max + 1), 3, padding=1),
        )
        self.register_buffer(
            "reg_bins", torch.arange(2 * self.reg_max + 1, dtype=torch.float32),
        )
        self.centerness_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 3, padding=1),
        )
        self.embed_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, embed_dim, 1),
        )

    def forward(self, feat):
        vis = self.vis_conv(feat)
        dist_logits = self.bbox_head(vis)
        batch, _, height, width = dist_logits.shape
        dist = dist_logits.reshape(batch, 4, 2 * self.reg_max + 1, height, width)
        prob = F.softmax(dist, dim=2)
        reg_pred = (prob * self.reg_bins.view(1, 1, -1, 1, 1)).sum(dim=2)
        return (
            reg_pred, dist_logits, self.cls_head(vis),
            self.centerness_head(vis), self.embed_head(vis),
        )


class SimFPN(nn.Module):
    def __init__(self, in_channels, num_layers=4, feat_dim=256):
        super().__init__()
        self.num_layers = num_layers
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, feat_dim, 1),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_layers)
        ])
        self.smooth_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        laterals = [conv(feat.float()) for feat, conv in zip(feats, self.lateral_convs)]
        fused = laterals[-1]
        for lateral in reversed(laterals[:-1]):
            fused = fused + lateral
        return self.smooth_conv(fused)


class LbotDetector(nn.Module):
    def __init__(self, variant="small", num_classes=1, embed_dim=8,
                 unfreeze_blocks=0, img_size=960, num_inter_layers=4,
                 feat_dim=256, device="auto", dtype="auto", backbone_path=None,
                 revision=None, cache_dir=None, local_files_only=False,
                 backbone_state_dict=None):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        self.img_size = img_size
        self.num_inter_layers = num_inter_layers
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_p2 = True
        self.variant = variant

        device = str(device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _resolve_dtype(dtype, device)
        if backbone_state_dict is None:
            self.backbone, backbone_embed_dim = load_pretrained_backbone(
                backbone_path, variant=variant, device=device, dtype=dtype,
                revision=revision, cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            config_variant = _normalize_variant(variant)
            cfg = load_config(PRETRAINED_VARIANTS[config_variant]["config_file"])
            self.backbone, backbone_embed_dim = build_backbone_from_cfg(cfg)
            self.backbone = self.backbone.to(device=device, dtype=dtype).eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        self.patch_size = self.backbone.patch_size

        if img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={self.patch_size}"
            )
        if unfreeze_blocks > 0:
            self.backbone.float()
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad_(True)

        step = len(self.backbone.blocks) // num_inter_layers
        self.intermediate_indices = [step * (i + 1) - 1 for i in range(num_inter_layers)]
        self.fpn = SimFPN(backbone_embed_dim, num_inter_layers, feat_dim)
        self.det_head = DetHead(feat_dim, feat_dim, num_classes, embed_dim)
        self.p2_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.p2_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.p2_head = DetHead(feat_dim, feat_dim, num_classes, embed_dim)

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        backbone_dtype = next(self.backbone.parameters()).dtype
        inter_feats = self.backbone.get_intermediate_layers(
            pixel_values.to(dtype=backbone_dtype), n=self.intermediate_indices,
            reshape=True, norm=True,
        )
        fused = self.fpn(inter_feats)
        p2_feat = self.p2_conv(fused) + self.p2_stem(pixel_values.to(dtype=fused.dtype))
        head_outputs = (self.det_head(fused), self.p2_head(p2_feat))
        reg_pred, reg_dist, cls_score, centerness, embed_map = (
            [output[index] for output in head_outputs] for index in range(5)
        )
        height, width = fused.shape[-2:]
        valid_mask = torch.ones(
            batch, height, width, dtype=torch.bool, device=fused.device,
        )
        return reg_pred, reg_dist, cls_score, centerness, valid_mask, embed_map
