# 数据许可与来源

## SODA-A

- 数据集：SODA-A（SODA: A Large-scale Benchmark for Small Object Detection）
- 官方主页：https://shaunyuan22.github.io/SODA/
- 官方代码仓库：https://github.com/shaunyuan22/SODA
- 许可：CC BY-NC 4.0，仅限学术非商业使用
- 许可全文：https://creativecommons.org/licenses/by-nc/4.0/

本仓库 `data_samples/` 中的 10 张图像和标签来自 SODA-A 验证集，保留原始文件名、图像尺寸和归一化坐标。图像版权与数据许可归原数据集权利人所有，不因放入本仓库而改变。

不得将这些样例用于商业用途。再次分发时请保留本文件、`data_samples/manifest.csv` 中的来源信息以及官方许可链接。

数据仅作样例，小目标训练效果不佳，请按格式自行替换数据集训练。

## 水平框转换说明

SODA-A 原始标注包含旋转框和 ignore 区域。本仓库使用的是既有 YOLO 转换结果，仅含水平框；类别 9 的 ignore 区域未保留。因此，本仓库样例不能替代 SODA-A 官方 OBB 评测协议，也不应与官方旋转框结果直接比较。
