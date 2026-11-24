# 基于 LingBot-Vision 的通用目标检测

面向自定义 YOLO 数据集的目标检测训练、评估和整图推理。模型使用 LingBot-Vision ViT-S/16 主干，融合多个中间层特征，并通过 stride-8 的 P2 分支增强小目标检测。

## 环境

需要 Python 3.10 或更高版本，训练建议使用 CUDA GPU。

```bash
python -m pip install -r requirements.txt
```

首次训练会下载 ViT-S/16 预训练主干，也可通过 `LBOT_BACKBONE_PATH` 指定本地权重。训练生成的 checkpoint 包含完整主干，评估和推理无需再次下载。

## 数据

仓库默认数据根目录为 `data_samples/`，训练和评估使用固定 YOLO 目录结构：

```text
data_samples/
├── classes.txt
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

标签格式为 `class_id x_center y_center width height`，坐标相对于原图归一化。类别数和类别顺序由 `classes.txt` 中的非空唯一类别名决定。自定义数据集保持相同结构，并通过 `--data-root` 和 `--classes` 指定数据根目录与类别文件。

`data_samples/` 包含 10 张样例图片，固定划分为 8 张训练图片和 2 张验证图片，仅用于验证训练、评估和推理链路。

训练、评估和推理统一读取完整原图：保持长宽比缩放到 960×960 画布内，小图不放大，剩余区域填充 114 灰度值。

## 训练

默认使用 `data_samples`，从冻结主干阶段开始训练，输出到 `runs/train_det/`：

```bash
python train_det_fast.py
```

快速检查训练链路：

```bash
python train_det_fast.py --epochs 1 --batch-size 1 --grad-accum 1 --device cpu
```

两阶段训练示例：先冻结主干训练，再解冻全部 block 微调。

```bash
python train_det_fast.py \
  --data-root data_samples --classes data_samples/classes.txt \
  --output-dir runs/frozen --stage frozen --device cuda

python train_det_fast.py \
  --data-root data_samples --classes data_samples/classes.txt \
  --output-dir runs/finetune --stage finetune \
  --resume runs/frozen/best.pth --device cuda
```

冻结阶段默认训练 600 轮；微调阶段固定训练 150 轮。`--resume` 只加载模型权重，不恢复优化器状态。训练输出 `best.pth`，早停时另有 `last.pth`。仓库不包含训练权重，评估和推理前需先训练生成 checkpoint。

## 评估

```bash
python evaluate.py \
  --weights runs/finetune/best.pth \
  --classes data_samples/classes.txt \
  --data-root data_samples --split val \
  --output-json runs/finetune/metrics.json --device cuda
```

评估输出 `mAP@0.5`、`mAP@0.5:0.95`、recall 以及按类别和尺寸桶统计的指标，并写入 JSON 文件。默认使用 0.50～0.95 的十个 IoU 阈值、0.05 置信度阈值和 0.50 NMS 阈值。

## 推理

```bash
python infer.py \
  --weights runs/finetune/best.pth \
  --classes data_samples/classes.txt \
  --source data_samples/images/val \
  --output-dir outputs/samples --device cuda
```

`--source` 支持单张图片或图片目录。推理输出 `predictions.jsonl` 和 `visualizations/`，预测框映射回原图像素坐标；使用 `--no-visualize` 可关闭可视化输出。

## 验证

```bash
python tools/check_repo.py
python -m unittest discover -s tests -v
```

## 许可

原创代码采用 MIT，见 [`LICENSE`](LICENSE)。LingBot-Vision 上游包采用 Apache-2.0，来源见 [`NOTICE.md`](NOTICE.md)。