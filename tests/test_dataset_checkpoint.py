import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import torch

import model
import train_det_fast
from config import ROOT
from runtime import load_categories, load_preprocessing
from train_det_fast import (
    DetDataset,
    _load_model_weights,
    collate_fn,
    parse_args,
    training_paths,
)


_PREPROCESSING = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "pad": 114,
    "image_size": 960,
}


class TrainingPathTests(unittest.TestCase):
    def test_class_file_accepts_arbitrary_category_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            classes = Path(tmp) / "classes.txt"
            expected = ["cat", "dog", "bird"]
            classes.write_text("\n".join(expected) + "\n", encoding="utf-8")
            self.assertEqual(load_categories(classes), expected)

    def test_training_entry_uses_class_file_category_count(self):
        class TrainingStarted(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            categories = ["cat", "dog", "bird"]
            (root / "classes.txt").write_text(
                "\n".join(categories) + "\n", encoding="utf-8"
            )
            args = train_det_fast.parse_args([
                "--data-root", str(root), "--device", "cpu", "--epochs", "1",
            ])
            captured = {}

            def stop_before_backbone_load(**kwargs):
                captured.update(kwargs)
                raise TrainingStarted

            with mock.patch.object(train_det_fast, "parse_args", return_value=args), \
                    mock.patch.object(model, "LbotDetector", side_effect=stop_before_backbone_load), \
                    mock.patch("builtins.print"):
                with self.assertRaises(TrainingStarted):
                    train_det_fast.main()

            self.assertEqual(captured["num_classes"], len(categories))

    def test_bundled_yolo_layout_is_ready_for_training(self):
        root = ROOT / "data_samples"
        args = parse_args(["--data-root", str(root)])
        self.assertEqual(Path(args.classes), root / "classes.txt")
        paths = training_paths(args.data_root)
        self.assertEqual(Path(paths["train_images"]), root / "images" / "train")
        self.assertEqual(Path(paths["train_labels"]), root / "labels" / "train")
        self.assertEqual(Path(paths["val_images"]), root / "images" / "val")
        self.assertEqual(Path(paths["val_labels"]), root / "labels" / "val")
        self.assertEqual(len(list((root / "images" / "train").glob("*.jpg"))), 8)
        self.assertEqual(len(list((root / "images" / "val").glob("*.jpg"))), 2)


class DatasetTests(unittest.TestCase):
    def test_training_uses_full_image_without_cropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (64, 32), (10, 20, 30)).save(images / "a.png")
            (labels / "a.txt").write_text(
                "0 0.25 0.5 0.2 0.2\n1 0.75 0.5 0.2 0.2\n", encoding="utf-8"
            )
            classes = root / "classes.txt"
            classes.write_text("left\nright\n", encoding="utf-8")
            ds = DetDataset(str(images), str(labels), str(classes),
                            img_size=32, patch_size=16, augment=False)
            item = ds[0]
            self.assertEqual(tuple(item["pixel_values"].shape), (3, 32, 32))
            self.assertEqual(item["num_boxes"], 2)
            self.assertEqual(item["target_labels"].tolist(), [0, 1])
            self.assertEqual(int(item["valid_mask"].sum()), 2)
            torch.testing.assert_close(item["target_boxes"], torch.tensor([
                [0.15, 0.4, 0.35, 0.6],
                [0.65, 0.4, 0.85, 0.6],
            ]))

    def test_boxes_crossing_image_boundary_are_clipped_not_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (32, 32)).save(images / "a.png")
            (labels / "a.txt").write_text("0 0.05 0.5 0.2 0.2\n", encoding="utf-8")
            classes = root / "classes.txt"
            classes.write_text("object\n", encoding="utf-8")
            ds = DetDataset(str(images), str(labels), str(classes),
                            img_size=32, patch_size=16, augment=False)
            torch.testing.assert_close(ds[0]["target_boxes"], torch.tensor([[0.0, 0.4, 0.15, 0.6]]))

    def test_collate_keeps_all_annotations_and_pads_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            classes = root / "classes.txt"
            classes.write_text("object\n", encoding="utf-8")
            for name, count in (("a", 1), ("b", 1201)):
                Image.new("RGB", (16, 16)).save(images / f"{name}.png")
                (labels / f"{name}.txt").write_text(
                    "0 0.5 0.5 0.1 0.1\n" * count, encoding="utf-8"
                )
            ds = DetDataset(str(images), str(labels), str(classes),
                            img_size=16, patch_size=16, augment=False)
            batch = collate_fn([ds[0], ds[1]])
            self.assertEqual(tuple(batch["target_boxes"].shape), (2, 1201, 4))
            self.assertEqual(int((batch["target_labels"][0] >= 0).sum()), 1)
            self.assertEqual(int((batch["target_labels"][1] >= 0).sum()), 1201)
            self.assertEqual(batch["target_labels"][0, -1].item(), -1)

    def test_invalid_class_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (32, 32)).save(images / "a.png")
            (labels / "a.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            classes = root / "classes.txt"
            classes.write_text("object\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class id"):
                DetDataset(str(images), str(labels), str(classes), img_size=32)

    def test_missing_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (32, 32)).save(images / "a.png")
            classes = root / "classes.txt"
            classes.write_text("object\n", encoding="utf-8")
            with self.assertRaisesRegex((ValueError, FileNotFoundError), "label"):
                DetDataset(str(images), str(labels), str(classes), img_size=32)


class _ModuleStub:
    def __init__(self):
        self.loaded = False

    def load_state_dict(self, state, strict=True):
        self.loaded = True


class _ModelStub:
    use_p2 = True
    det_head = type("Head", (), {"reg_max": 8, "embed_dim": 8})()
    num_classes = 1
    img_size = 960
    variant = "small"
    num_inter_layers = 4

    def __init__(self):
        self.fpn = _ModuleStub()
        self.det_head = _ModuleStub()
        self.det_head.reg_max = 8
        self.det_head.embed_dim = 8
        self.p2_conv = _ModuleStub()
        self.p2_head = _ModuleStub()
        self.p2_stem = _ModuleStub()


class CheckpointTests(unittest.TestCase):
    def test_preprocessing_validation(self):
        self.assertEqual(load_preprocessing({"preprocessing": _PREPROCESSING})["pad"], 114)
        saved = {**_PREPROCESSING, "mean": [0.5, 0.5, 0.5],
                 "std": [0.25, 0.25, 0.25], "pad": 0, "image_size": 64}
        self.assertEqual(load_preprocessing({"preprocessing": saved})["image_size"], 64)
        with self.assertRaisesRegex(ValueError, "std"):
            load_preprocessing({"preprocessing": {**_PREPROCESSING, "std": [1, 0, 1]}})
        with self.assertRaisesRegex(ValueError, "pad"):
            load_preprocessing({"preprocessing": {**_PREPROCESSING, "pad": 256}})
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            load_preprocessing({})
        with self.assertRaisesRegex(ValueError, "image_size"):
            load_preprocessing({"preprocessing": {**_PREPROCESSING, "image_size": 65}})

    def test_incomplete_p2_checkpoint_is_rejected(self):
        ckpt = {
            "categories": ["object"],
            "architecture": {
                "variant": "small", "use_p2": True, "reg_max": 8,
                "num_classes": 1, "embed_dim": 8, "img_size": 960,
                "num_inter_layers": 4,
            },
            "fpn": {}, "det_head": {}, "p2_conv": {}, "p2_head": {},
        }
        with self.assertRaisesRegex(ValueError, "p2_stem"):
            _load_model_weights(_ModelStub(), ckpt)

    def test_category_mismatch_is_rejected(self):
        ckpt = {"categories": ["other"], "fpn": {}, "det_head": {}}
        with self.assertRaisesRegex(ValueError, "categories"):
            _load_model_weights(_ModelStub(), ckpt, expected_categories=["object"])


if __name__ == "__main__":
    unittest.main()
