import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import evaluate
import infer
import model
from runtime import load_detector


class _DummyBackbone(nn.Module):
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
        self.proj = nn.Conv2d(3, 384, 1)

    def get_intermediate_layers(self, x, n, reshape=True, norm=True):
        feat = self.proj(F.avg_pool2d(x, 16))
        return tuple(feat for _ in n)


class SyntheticPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.classes = [f"c{i}" for i in range(3)]
        (self.root / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        Image.new("RGB", (128, 96), (80, 120, 160)).save(self.root / "image.png")
        labels = self.root / "labels"
        labels.mkdir()
        (labels / "image.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        with mock.patch.object(model, "load_pretrained_backbone",
                               return_value=(_DummyBackbone(), 384)):
            detector = model.LbotDetector(
                variant="small", num_classes=len(self.classes), embed_dim=8,
                unfreeze_blocks=0, img_size=64,
                device="cpu", dtype=torch.float32,
            )
        self.weights = self.root / "synthetic.pth"
        torch.save({
            "fpn": detector.fpn.state_dict(), "det_head": detector.det_head.state_dict(),
            "p2_conv": detector.p2_conv.state_dict(), "p2_head": detector.p2_head.state_dict(),
            "p2_stem": detector.p2_stem.state_dict(), "backbone": detector.backbone.state_dict(),
            "categories": self.classes,
            "architecture": {"variant": "small", "use_p2": True, "reg_max": 8,
                             "num_classes": len(self.classes), "embed_dim": 8, "img_size": 64,
                             "num_inter_layers": 4},
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                "pad": 114, "image_size": 64,
            },
        }, self.weights)
        self.patch = mock.patch.object(
            model, "build_backbone_from_cfg",
            return_value=(_DummyBackbone(), 384),
        )
        self.patch.start()
        self.tmp_cleanup = self.tmp.cleanup

    def tearDown(self):
        self.patch.stop()
        self.tmp_cleanup()

    def test_full_image_inference_writes_original_coordinates_offline(self):
        args = infer.parse_args([
            "--weights", str(self.weights), "--source", str(self.root / "image.png"),
            "--classes", str(self.root / "classes.txt"),
            "--output-dir", str(self.root / "out"), "--device", "cpu",
            "--no-visualize",
        ])
        with mock.patch.object(
            model, "load_pretrained_backbone",
            side_effect=AssertionError("full checkpoints must load offline"),
        ):
            self.assertEqual(infer.run(args), 0)
        records = [json.loads(line) for line in
                   (self.root / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in records:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            self.assertTrue(0 <= x1 <= x2 <= 128)
            self.assertTrue(0 <= y1 <= y2 <= 96)

    def test_checkpoint_image_size_mismatch_is_rejected_before_model_build(self):
        ckpt = torch.load(self.weights, map_location="cpu", weights_only=True)
        ckpt["architecture"]["img_size"] = 128
        torch.save(ckpt, self.weights)
        with mock.patch.object(
            model, "build_backbone_from_cfg",
            side_effect=AssertionError("invalid metadata must fail before model construction"),
        ):
            with self.assertRaisesRegex(ValueError, "img_size"):
                load_detector(self.weights, self.classes, device="cpu")

    def test_checkpoint_num_classes_mismatch_is_rejected(self):
        ckpt = torch.load(self.weights, map_location="cpu", weights_only=True)
        ckpt["architecture"]["num_classes"] = len(self.classes) + 1
        torch.save(ckpt, self.weights)
        with self.assertRaisesRegex(ValueError, "num_classes"):
            load_detector(self.weights, self.classes, device="cpu")

    def test_standalone_evaluation_uses_complete_iou_range(self):
        args = evaluate.parse_args([
            "--weights", str(self.weights), "--classes", str(self.root / "classes.txt"),
            "--image-dir", str(self.root), "--label-dir", str(self.root / "labels"),
            "--output-json", str(self.root / "metrics.json"), "--device", "cpu",
        ])
        with mock.patch("builtins.print"):
            self.assertEqual(evaluate.run(args), 0)
        payload = json.loads((self.root / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("mAP@0.5", payload["results"])
        self.assertIn("mAP@0.5:0.95", payload["results"])
        self.assertEqual(payload["protocol"]["iou_thresholds"][-1], 0.95)
        self.assertEqual(payload["protocol"]["image_size"], 64)
        self.assertEqual(payload["protocol"]["resize_mode"], "full-image aspect-preserving letterbox")
        self.assertEqual(payload["evaluation"]["n_images"], 1)


if __name__ == "__main__":
    unittest.main()
