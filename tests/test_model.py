import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

import model
from third_party.lingbot_vision.lingbot_vision.loader import _backbone_state, load_backbone
from train_det_fast import _load_model_weights


class _DummyBackbone(nn.Module):
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
        self.proj = nn.Conv2d(3, 384, 1)

    def get_intermediate_layers(self, x, n, reshape=True, norm=True):
        feat = self.proj(F.avg_pool2d(x, 16))
        return tuple(feat + (i + 1) * 0.001 for i in n)


class ModelTests(unittest.TestCase):
    def _build(self, num_classes=4):
        with mock.patch.object(model, "load_pretrained_backbone",
                               return_value=(_DummyBackbone(), 384)):
            return model.LbotDetector(
                variant="small", num_classes=num_classes, embed_dim=8,
                unfreeze_blocks=0, img_size=64,
                device="cpu", dtype=torch.float32,
            )

    def test_p2_forward_shapes(self):
        model = self._build().eval()
        with torch.no_grad():
            reg, dist, cls, cent, vm, embed = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(len(reg), 2)
        self.assertEqual(tuple(cls[0].shape), (2, 4, 4, 4))
        self.assertEqual(tuple(cls[1].shape), (2, 4, 8, 8))
        self.assertEqual(tuple(dist[0].shape), (2, 68, 4, 4))
        self.assertEqual(tuple(cent[1].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(embed[1].shape), (2, 8, 8, 8))
        self.assertEqual(tuple(vm.shape), (2, 4, 4))

    def test_invalid_num_classes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_classes"):
            self._build(0)

    def test_modular_checkpoint_roundtrip(self):
        model = self._build().eval()
        ckpt = {
            "fpn": model.fpn.state_dict(),
            "det_head": model.det_head.state_dict(),
            "p2_conv": model.p2_conv.state_dict(),
            "p2_head": model.p2_head.state_dict(),
            "p2_stem": model.p2_stem.state_dict(),
            "backbone": model.backbone.state_dict(),
            "categories": [f"c{i}" for i in range(4)],
            "architecture": {
                "variant": "small", "use_p2": True, "reg_max": 8,
                "num_classes": 4, "embed_dim": 8, "img_size": 64,
                "num_inter_layers": 4,
            },
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "pad": 114, "image_size": 64,
            },
        }
        restored = self._build().eval()
        _load_model_weights(restored, ckpt, expected_categories=ckpt["categories"])
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = model(x)
            actual = restored(x)
        for a, b in zip(expected, actual):
            if isinstance(a, list):
                for aa, bb in zip(a, b):
                    torch.testing.assert_close(aa, bb)
            else:
                torch.testing.assert_close(a, b)
        bad = {**ckpt, "fpn": {**ckpt["fpn"], "unexpected": torch.tensor(1.)}}
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            _load_model_weights(restored, bad, expected_categories=ckpt["categories"])
        bad = {**ckpt, "fpn": dict(list(ckpt["fpn"].items())[1:])}
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            _load_model_weights(restored, bad, expected_categories=ckpt["categories"])


class BackboneLoaderTests(unittest.TestCase):
    def test_checkpoint_wrappers_and_strict_loading(self):
        state = {"weight": torch.zeros(1, 1), "bias": torch.zeros(1)}
        for wrapper in ("teacher", "model_state", "state_dict", "model", "backbone"):
            with self.subTest(wrapper=wrapper):
                self.assertEqual(_backbone_state({wrapper: state}), state)
        self.assertEqual(_backbone_state({"backbone." + k: v for k, v in state.items()}), state)
        self.assertEqual(_backbone_state({"_orig_mod." + k: v for k, v in state.items()}), state)
        with mock.patch(
            "third_party.lingbot_vision.lingbot_vision.loader.build_backbone_from_cfg",
            return_value=(nn.Linear(1, 1), 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_backbone(None, {"weight": state["weight"]}, device="cpu")
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                load_backbone(None, {**state, "extra": torch.tensor(1.)}, device="cpu")


if __name__ == "__main__":
    unittest.main()
