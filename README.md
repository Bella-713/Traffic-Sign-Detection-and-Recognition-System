# 交通标志检测与识别系统

基于 **YOLOv8** 与 **GTSDB** 数据集的交通标志检测系统，支持对 **5 类** 常见交通标志进行实时检测与识别。

本项目作为计算机视觉课程作业，完整实现了数据集预处理、模型训练（含基准与改进策略对比）、多维度评估与可视化分析的全流程。

## 项目概述

- **任务**：目标检测（Object Detection）—— 在图像中定位交通标志并识别其类别
- **模型**：Ultralytics YOLOv8s（Small 版本，兼顾速度与精度）
- **数据集**：GTSDB（German Traffic Sign Detection Benchmark），原始含 43 类交通标志
- **类别映射**：将原始 43 类合并为 5 类 —— 限速(speed_limit)、禁止通行(no_entry)、方向指引(direction)、人行横道(crosswalk)、停车让行(stop_yield)
- **训练策略**：提供 **基准模型 (Baseline)** 与 **改进模型 (Improved)** 两组训练管线，便于对比分析
- **评估维度**：mAP、Precision/Recall、混淆矩阵、损失曲线、不同场景（明亮/正常/低光/模糊）性能差异、误检/漏检案例分析

## 项目成员

- 王虹力、许婧妍：环境搭建、模型训练、模型测试以及不同条件检测性能对比分析
- 唐敏慧、徐文颖、周轩瑶：实验报告撰写及数据整理

## 环境要求

| 依赖 | 版本/说明 |
|------|-----------|
| Python | 3.10 |
| PyTorch | ≥ 2.0.0（CUDA 12.6 版） |
| CUDA | 12.6 + cuDNN（for CUDA 12.x） |
| GPU | NVIDIA GPU，显存 ≥ 4GB |
| 操作系统 | Windows 11（建议） |

## 安装步骤

### 1. 克隆项目

```bash
git clone <项目地址>
cd traffic_sign_project
```

### 2. 创建虚拟环境

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. 安装 PyTorch（CUDA 版）

确保与本地 CUDA 版本匹配：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> 如使用其他 CUDA 版本，请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应安装命令。

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

主要依赖见 [requirements.txt](requirements.txt)：
- `ultralytics` — YOLOv8 训练与推理框架
- `opencv-python` — 图像处理
- `matplotlib` / `seaborn` — 可视化与图表生成
- `scikit-learn` / `pandas` — 数据分析与指标计算
- `albumentations` — 数据增强（改进策略使用）
- `tqdm` — 进度条

## 数据集准备

1. 从 Kaggle 下载 GTSDB 数据集：
   https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb

2. 将下载的数据集解压后放入项目根目录下的 `dataset/` 文件夹，目录结构如下：

   ```
   dataset/
   ├── gt.txt                 # 标注文件 (图片名;x1;y1;x2;y2;类别ID)
   ├── TrainIJCNN2013/        # 训练图片（.ppm 格式）
   └── TestIJCNN2013/         # 测试图片
   ```

## 使用说明

### 1. 完整训练（数据预处理 + 基准训练 + 改进训练）

一键执行数据预处理、基准模型训练和改进策略训练：

```bash
python code\train.py
```

该脚本自动完成：

| 步骤 | 说明 |
|------|------|
| **数据预处理** | 读取 GTSDB 标注 → 43 类映射为 5 类 → 转换为 YOLO 格式（归一化 cx,cy,w,h）→ 按 8:2 划分训练/验证集 |
| **基准模型训练** | 使用 YOLOv8s 默认超参数训练，保存权重至 `runs/baseline/` |
| **改进模型训练** | 在基准基础上启用强数据增强（Mosaic、MixUp、Copy-Paste）、余弦退火学习率调度、标签平滑、更长的预热与早停耐心，保存权重至 `runs/improved/` |

**改进策略亮点：**

- **数据增强**：饱和度/明度抖动、旋转 ±15°、缩放 ±60%、Mosaic 拼接、MixUp 混合、Copy-Paste 小目标增强
- **学习率调度**：余弦退火（Cosine Annealing）+ 5 轮预热
- **正则化**：标签平滑（label_smoothing=0.05），防止过拟合
- **精调策略**：最后 15 轮关闭 Mosaic 增强，稳定收敛

#### 训练参数说明

```bash
python code\train.py --dataset ../dataset   # 指定数据集路径
                     --output ../dataset/yolo_format  # YOLO 格式输出路径
                     --epochs 100           # 训练轮数
                     --batch 16             # Batch size
                     --model yolov8s.pt     # 模型 (n/s/m/l/x)
```

### 2. 单独训练 Baseline

若只需训练基准模型（不训练改进模型）：

```bash
python code\train_baseline.py
```

### 3. 测试推理

使用训练好的模型对单张图片进行检测推理：

```bash
python -c "from model import TrafficSignDetector; d = TrafficSignDetector(); r = d.predict('path/to/image.jpg', 'results/baseline_best.pt'); r[0].show()"
```

或使用 Jupyter Notebook 交互式演示：

```bash
jupyter notebook code/demo.ipynb
```

### 4. 模型评估与可视化

```bash
python code\eval.py
```

评估脚本依次执行以下 6 个模块，所有结果保存至 `results/` 目录：

| # | 模块 | 输出文件 | 说明 |
|---|------|----------|------|
| 1 | 模型性能对比 | `comparison.json` | Baseline vs Improved 的 mAP50 / mAP50:95 / Precision / Recall / F1 |
| 2 | 损失曲线 | `loss_curve_{baseline,improved}.png` | 训练过程中的 Loss、Precision、Recall 曲线 |
| 3 | 混淆矩阵 | `confusion_matrix.png` | 5 类别的预测 vs 真实类别混淆矩阵 |
| 4 | 场景对比分析 | `comparison_table.png` | 按亮度/模糊度分组评估（明亮/正常/低光/模糊） |
| 5 | 误检漏检分析 | `false_detection_analysis.png` | 可视化 TP/FP/FN 案例及统计 |
| 6 | 评估报告 | `eval_report.txt` | 汇总全部数值指标 |

### 5. 单图预测

```python
from model import TrafficSignDetector

detector = TrafficSignDetector()
results = detector.predict("path/to/image.jpg", "results/baseline_best.pt", conf=0.25)
results[0].show()  # 显示检测结果
```

### 6. 模型导出（ONNX）

```python
from model import TrafficSignDetector

detector = TrafficSignDetector()
detector.export_onnx("results/baseline_best.pt", "results/model.onnx")
```

## 实验结果

### 性能指标对比

| 指标 | 基准模型 (Baseline) | 改进模型 (Improved) | 提升幅度 |
|------|:-------------------:|:-------------------:|:--------:|
| mAP@0.5 | **86.85%** | **90.15%** | **+3.30%** |
| mAP@0.5:0.95 | 72.11% | **72.28%** | +0.17% |
| Precision (精确率) | **93.70%** | 92.45% | -1.25% |
| Recall (召回率) | 79.01% | **88.95%** | **+9.94%** |
| F1-score | 85.73% | **90.67%** | **+4.94%** |

> 改进策略在保持高精确率的同时，大幅提升了召回率（+9.94%），
> 表明强数据增强和余弦退火调度有效提升了模型的泛化能力，
> 减少了漏检情况，综合 F1-score 提升 4.94%。

### 场景适应性分析

| 场景 | mAP@0.5 | 说明 |
|------|:-------:|------|
| 明亮 (Bright) | — | 高光照条件下的检测表现 |
| 正常 (Normal) | — | 常规光照下的检测表现 |
| 低光 (Dark) | — | 夜间/弱光环境下的检测表现 |
| 模糊 (Blurry) | — | 运动模糊/失焦情况下的检测表现 |

> 各场景详细数据由 `eval.py` 的场景分析模块自动生成，
> 结果详见 `results/comparison_table.png`。

### 评估输出文件

| 文件 | 内容 |
|------|------|
| `results/loss_curve.png` | 训练损失曲线 |
| `results/confusion_matrix.png` | 混淆矩阵热力图 |
| `results/comparison_table.png` | 多场景性能对比表 |
| `results/false_detection_analysis.png` | 误检/漏检案例分析 |
| `results/eval_report.txt` | 数值评估报告 |
| `results/comparison.json` | 结构化对比数据 |

## 类别说明

| ID | 类别 | 英文标识 | 说明 | 对应 GTSDB 原始类别 |
|----|------|----------|------|---------------------|
| 0 | 限速标志 | speed_limit | 限速 30/50/60/80/100/120 | 0,1,2,3,4,5,7,8 |
| 1 | 禁止通行 | no_entry | 禁止驶入标志 | 17 |
| 2 | 方向指示 | direction | 直行/左转/右转等强制方向 | 33,34,35,36,37,38,39 |
| 3 | 人行横道 | crosswalk | 人行横道警告 | 27 |
| 4 | 停车让行 | stop_yield | 停止/让行标志 | 13,14 |

## 项目结构

```
traffic_sign_project/                          # 项目根目录
├── code/                                      # 源代码
│   ├── README.md                              # 项目文档（本文件）
│   ├── requirements.txt                       # Python 依赖清单
│   ├── data_utils.py                          # 数据处理工具
│   │   └── GTSDB 加载、43→5 映射、YOLO 格式转换、场景分类
│   ├── model.py                               # 模型定义
│   │   └── TrafficSignDetector 类（训练/评估/预测/ONNX导出）
│   ├── train.py                               # 训练脚本（Baseline + Improved）
│   ├── train_baseline.py                      # 单独训练 Baseline
│   ├── eval.py                                # 评估脚本（6 大评估模块）
│   └── demo.ipynb                             # Jupyter 演示 Notebook
├── dataset/                                   # 数据集（需自行下载）
│   ├── gt.txt
│   ├── TrainIJCNN2013/
│   ├── TestIJCNN2013/
│   └── yolo_format/                           # 预处理后的 YOLO 格式数据
│       ├── data.yaml                          # YOLO 数据集配置文件
│       ├── images/{train,val}/
│       └── labels/{train,val}/
├── runs/                                      # 训练产出
│   ├── baseline/                              # 基准模型结果
│   │   ├── weights/{best,last}.pt
│   │   ├── results.csv
│   │   └── *.jpg (训练/验证样本图)
│   └── improved/                              # 改进模型结果
│       └── ...（同上结构）
├── results/                                   # 评估输出
│   ├── comparison.json                        # 性能对比数据
│   ├── loss_curve.png                         # 损失曲线图
│   ├── confusion_matrix.png                   # 混淆矩阵
│   ├── comparison_table.png                   # 场景对比表
│   ├── false_detection_analysis.png           # 误检漏检分析
│   └── eval_report.txt                        # 评估报告
└── contribution.txt                           # 分工声明
```

## 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| mAP@0.5 | ≥ 75% | 在 IoU=0.5 下的平均精度均值 |
| mAP@0.5:0.95 | — | 在 IoU 0.5~0.95（步长 0.05）下的平均精度均值 |
| Precision | — | 检测结果的准确率（TP/(TP+FP)） |
| Recall | — | 检测结果的召回率（TP/(TP+FN)） |
| 场景适应性 | — | 分别在明亮/正常/低光/模糊场景下的表现对比 |
