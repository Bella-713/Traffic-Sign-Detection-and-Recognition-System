# 交通标志检测与识别系统

基于 YOLOv8 和 GTSDB 数据集的交通标志检测系统，支持 5 类交通标志识别。

## 项目成员

- 姓名1: [角色/分工]
- 姓名2: [角色/分工]

## 环境要求

- Python 3.10
- CUDA 12.6 + cuDNN (for CUDA 12.x)
- NVIDIA GPU (显存 ≥ 4GB)
- Windows 11

## 安装步骤

1. 创建虚拟环境:
```bash
python -m venv venv
venv\Scripts\activate
```

2. 安装 PyTorch（CUDA 版）:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

3. 安装其他依赖:
```bash
pip install -r requirements.txt
```

## 数据集准备

1. 从 Kaggle 下载 GTSDB 数据集:
   https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb

2. 将数据集放入 `dataset/` 目录

## 使用说明

### 1. 数据预处理 + 训练

```bash
python code\train.py
```

会自动执行:
- GTSDB → YOLO 格式转换
- 43类 → 5类 映射
- 基准模型训练 (YOLOv8s)
- 改进策略训练 (增强版)

### 2. 评估和生成图表

```bash
python code\eval.py
```

会生成:
- `results/loss_curve.png` — 训练损失曲线
- `results/confusion_matrix.png` — 混淆矩阵
- `results/comparison_table.png` — 不同场景性能对比
- `results/eval_report.txt` — 评估报告

## 类别说明

| ID | 类别 | 说明 |
|----|------|------|
| 0 | speed_limit | 限速标志 (30/50/60/80/100/120) |
| 1 | no_entry | 禁止通行 |
| 2 | direction | 直行/转弯指示 |
| 3 | crosswalk | 人行横道 |
| 4 | stop_yield | 停车让行 |

## 项目结构

```
traffic_sign_project/
├── code/
│   ├── README.md
│   ├── requirements.txt
│   ├── data_utils.py    # 数据处理
│   ├── model.py         # 模型定义
│   ├── train.py         # 训练脚本
│   └── eval.py          # 评估脚本
├── dataset/             # 数据集
├── report/              # 实验报告
├── results/             # 结果图表
└── contribution.txt     # 分工声明
```

## 性能指标

- mAP@0.5 ≥ 75% (目标)
- 支持 5 类交通标志检测
- 不同场景（晴朗/夜间/模糊）对比分析
