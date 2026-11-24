import unittest
from unittest import mock

import torch

from train_det_fast import (
    EVAL_IOU_THRS,
    _ap_from_pr,
    _longest_side_pixels,
    _nms,
    compute_loss,
    evaluate,
)


class MetricAndNmsTests(unittest.TestCase):
    def test_coco_iou_range_is_complete(self):
        self.assertEqual(
            EVAL_IOU_THRS,
            tuple(round(0.5 + 0.05 * i, 2) for i in range(10)),
        )

    def test_pixel_box_sides_are_not_scaled_twice(self):
        boxes = torch.tensor([[100., 200., 200., 260.], [0., 0., 16., 32.]])
        self.assertEqual(_longest_side_pixels(boxes).tolist(), [100., 32.])

    def test_zero_positive_scale_has_zero_gradient_weight(self):
        pred = torch.zeros(1, 1, 1, 1)
        target_boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.2],
                                      [0.5, 0.5, 0.6, 0.6]]])
        target_labels = torch.tensor([[0, -1]])
        losses = [
            (torch.tensor(100.), torch.tensor(100.), *[torch.tensor(0.)] * 5, 0),
            (torch.tensor(2.), torch.tensor(2.), *[torch.tensor(0.)] * 5, 2),
        ]
        with mock.patch("train_det_fast._compute_loss_scale", side_effect=losses):
            total, cls_loss, *_ = compute_loss(
                [pred, pred], [pred, pred], [pred, pred], [pred, pred],
                target_boxes, target_labels, "cpu",
            )
        self.assertAlmostEqual(float(total), 2.0)
        self.assertAlmostEqual(float(cls_loss), 2.0)

    def test_scale_routing_uses_valid_content_and_preserves_ignore_boxes(self):
        pred = [torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 8, 8)]
        target_boxes = torch.tensor([[[0.1, 0.1, 0.7, 0.3],
                                      [0.5, 0.5, 0.75, 0.7]]])
        target_labels = torch.tensor([[0, 1]])
        calls = []

        def loss_scale(boxes, scores, centerness, embed, boxes_gt, labels, device, **kwargs):
            calls.append((labels.detach().clone(), kwargs["ignore_boxes"].detach().clone()))
            return (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.),
                    torch.tensor(0.), torch.tensor(0.), torch.tensor(0.),
                    torch.tensor(0.), 1)

        with mock.patch("train_det_fast._compute_loss_scale", side_effect=loss_scale):
            compute_loss(
                pred, pred, pred, pred, target_boxes, target_labels, "cpu",
                valid_mask=torch.ones(1, 2, 4, dtype=torch.bool),
            )
        torch.testing.assert_close(calls[0][0], torch.tensor([[0, -1]]))
        torch.testing.assert_close(calls[0][1], torch.tensor(
            [[[2.0, 2.0, 2.0, 2.0], [0.5, 0.5, 0.75, 0.7]]],
        ))
        torch.testing.assert_close(calls[1][0], torch.tensor([[-1, 1]]))
        torch.testing.assert_close(calls[1][1], torch.tensor(
            [[[0.1, 0.1, 0.7, 0.3], [2.0, 2.0, 2.0, 2.0]]],
        ))

    def test_all_point_ap_perfect_and_empty(self):
        self.assertAlmostEqual(_ap_from_pr([1.0, 1.0], [0.5, 1.0]), 1.0)
        self.assertEqual(_ap_from_pr([], []), 0.0)

    def test_nms_suppresses_only_overlapping_same_input(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.],
                              [20., 20., 30., 30.]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = _nms(boxes, scores, 0.5).tolist()
        self.assertEqual(keep, [0, 2])


class _EvalDataset:
    categories = ["object"]
    files = ["image.png"]

    def __init__(self, image_size, gt_box):
        self.image_size = image_size
        self.gt_box = gt_box

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "pixel_values": torch.zeros(3, 64, 64),
            "target_boxes": torch.zeros(1, 4),
            "target_labels": torch.full((1,), -1, dtype=torch.long),
            "valid_mask": torch.ones(4, 4, dtype=torch.bool),
            "src_idx": torch.tensor(0),
        }

    def img_wh(self, index):
        return self.image_size

    def load_gt(self, index):
        width, height = self.image_size
        x1, y1, x2, y2 = self.gt_box
        return [[x1 / width, y1 / height, x2 / width, y2 / height]], [0]


class _PerfectDetectionModel(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        reg = [torch.zeros(batch, 4, 4, 4), torch.zeros(batch, 4, 8, 8)]
        cls = [torch.full((batch, 1, 4, 4), -20.0),
               torch.full((batch, 1, 8, 8), -20.0)]
        cent = [torch.ones(batch, 1, 4, 4), torch.ones(batch, 1, 8, 8)]
        embed = [torch.zeros(batch, 8, 4, 4), torch.zeros(batch, 8, 8, 8)]
        for index, (grid_x, grid_y, distances) in enumerate(self.predictions[:batch]):
            reg[0][index, :, grid_y, grid_x] = torch.tensor(distances)
            cls[0][index, 0, grid_y, grid_x] = 20.0
        return reg, reg, cls, cent, torch.ones(batch, 4, 4, dtype=torch.bool), embed


class EvaluationRegressionTests(unittest.TestCase):
    def test_ground_truth_count_is_not_repeated_for_size_buckets(self):
        dataset = _EvalDataset((64, 64), (16, 16, 32, 32))
        model = _PerfectDetectionModel([(1, 1, (0.5, 0.5, 0.5, 0.5))])
        results, _, _ = evaluate(model, dataset, "cpu", eval_batch=1)
        self.assertAlmostEqual(results["mAP@0.5"], 1.0, places=5)
        self.assertAlmostEqual(results["recall@0.5"], 1.0)
        self.assertAlmostEqual(results["recall@0.5[small16-32px]"], 1.0)

    def test_full_image_nms_uses_original_image_pixels(self):
        dataset = _EvalDataset((64, 64), (16, 16, 32, 32))
        model = _PerfectDetectionModel([(1, 1, (0.5, 0.5, 0.5, 0.5))])
        original_nms = _nms
        calls = []

        def record_nms(boxes, scores, threshold):
            calls.append(boxes.detach().clone())
            return original_nms(boxes, scores, threshold)

        with mock.patch("runtime.nms", side_effect=record_nms):
            results, _, _ = evaluate(model, dataset, "cpu", eval_batch=1)
        torch.testing.assert_close(calls[0], torch.tensor([[16., 16., 32., 32.]]))
        self.assertAlmostEqual(results["mAP@0.5"], 1.0, places=5)
        self.assertAlmostEqual(results["recall@0.5"], 1.0)


if __name__ == "__main__":
    unittest.main()
