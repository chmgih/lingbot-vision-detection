"""Load the ViT-S/16 backbone used by the detector."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from .build import build_backbone_from_cfg


DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
PRETRAINED_VARIANTS = {
    "small": {
        "repo_id": "robbyant/lingbot-vision-vit-small",
        "config_file": "configs/lbot_vision_vits.yaml",
        "checkpoint_file": "model.pt",
    },
}
_VARIANT_ALIASES = {"s": "small", "vit_s": "small", "vits": "small", "small": "small"}


def _normalize_variant(variant="auto"):
    if variant is None or str(variant).lower() in ("auto", ""):
        return "small"
    key = str(variant).lower().replace("-", "_")
    if key not in _VARIANT_ALIASES:
        raise ValueError(f"unknown variant {variant!r}; expected small")
    return _VARIANT_ALIASES[key]


def _resolve_dtype(dtype, device):
    if dtype == "auto":
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    if isinstance(dtype, str):
        if dtype not in DTYPE_MAP:
            raise ValueError(f"unknown dtype {dtype!r}")
        return DTYPE_MAP[dtype]
    return dtype


def _packaged_config_path(config_file):
    path = Path(config_file).expanduser()
    if path.is_file():
        return path
    packaged = Path(__file__).parent / "configs" / path.name
    return packaged if packaged.is_file() else None


def resolve_pretrained_files(
    repo_id_or_path=None,
    variant="auto",
    *,
    config_file=None,
    checkpoint_file=None,
    cache_dir=None,
    revision=None,
    local_files_only=False,
):
    variant = _normalize_variant(variant)
    spec = PRETRAINED_VARIANTS[variant]
    repo_id_or_path = repo_id_or_path or spec["repo_id"]
    config_file = config_file or spec["config_file"]
    checkpoint_file = checkpoint_file or spec["checkpoint_file"]
    root = Path(repo_id_or_path).expanduser()
    if root.exists():
        if not root.is_dir():
            raise ValueError("repo_id_or_path must be a directory when using local files")
        config_path = root / config_file
        if not config_path.is_file():
            config_path = _packaged_config_path(config_file)
        checkpoint_path = root / checkpoint_file
    else:
        config_path = _packaged_config_path(config_file)
        if config_path is None:
            config_path = Path(hf_hub_download(
                repo_id_or_path, config_file, cache_dir=cache_dir,
                revision=revision, local_files_only=local_files_only,
            ))
        checkpoint_path = Path(hf_hub_download(
            repo_id_or_path, checkpoint_file, cache_dir=cache_dir,
            revision=revision, local_files_only=local_files_only,
        ))
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    return config_path, checkpoint_path


def load_pretrained_backbone(
    repo_id_or_path=None,
    variant="auto",
    *,
    device=None,
    dtype="auto",
    cache_dir=None,
    revision=None,
    local_files_only=False,
    config_file=None,
    checkpoint_file=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(dtype, device)
    config_path, checkpoint_path = resolve_pretrained_files(
        repo_id_or_path, variant, config_file=config_file,
        checkpoint_file=checkpoint_file, cache_dir=cache_dir,
        revision=revision, local_files_only=local_files_only,
    )
    return load_backbone(
        load_config(config_path), load_backbone_state(checkpoint_path),
        device=device, dtype=dtype,
    )


def load_config(config_file):
    config_path = _packaged_config_path(config_file)
    if config_path is None:
        raise FileNotFoundError(str(config_file))
    return OmegaConf.load(config_path)


def load_backbone_state(backbone_path, map_location="cpu"):
    backbone_path = Path(backbone_path)
    if not backbone_path.is_file():
        raise FileNotFoundError(str(backbone_path))
    return torch.load(str(backbone_path), map_location=map_location, weights_only=True)


def _backbone_state(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    cleaned = {key.replace("_orig_mod.", ""): value for key, value in ckpt.items()}
    prefixed = {
        key[len("backbone."):]: value for key, value in cleaned.items()
        if key.startswith("backbone.")
    }
    if prefixed:
        return prefixed
    for key in ("teacher", "model_state", "state_dict", "model", "backbone"):
        if isinstance(cleaned.get(key), dict):
            return cleaned[key]
    return cleaned


def load_backbone(cfg, ckpt, device="cuda", dtype=torch.float32):
    model, embed_dim = build_backbone_from_cfg(cfg)
    model.load_state_dict(_backbone_state(ckpt), strict=True)
    model = model.to(device).to(dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, embed_dim
