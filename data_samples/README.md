# 样例数据

包含 10 张样例图像及同名 YOLO 水平框标签，共 17,519 个实例，覆盖全部 9 类。仓库内固定划分为 8 张训练图片和 2 张验证图片，仅用于验证训练、评估和推理链路。

- `images/train/`、`images/val/`：按 YOLO 数据集约定划分的图片。
- `labels/train/`、`labels/val/`：对应的 YOLO 水平框标签，格式为 `class_id x_center y_center width height`，坐标相对于原图归一化。
- `classes.txt`：类别顺序。
- `manifest.csv`：图片/标签路径、划分、原文件名、尺寸、实例数、`tiny_lt16`、`median_longest_side_px`、`class_counts` 和 SHA-256。
