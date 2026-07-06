"""
交通标志检测 - 训练脚本
执行: (1) 数据预处理 (2) 基准训练 (3) 改进策略训练
"""

import os
import sys
import shutil
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'code'))

from data_utils import prepare_dataset
from model import TrafficSignDetector


def main():
    # 必须在 import torch 前设置，否则无效
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 初始化 CUDA（放在 main 里，避免 Windows DataLoader 子进程 re-import 时死锁）
    import torch
    if torch.cuda.is_available():
        _ = torch.zeros(1).cuda()
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    parser = argparse.ArgumentParser(description='交通标志检测训练')
    parser.add_argument('--dataset', type=str,
                        default=os.path.join(project_root, 'dataset'),
                        help='GTSDB 数据集根目录')
    parser.add_argument('--output', type=str,
                        default=os.path.join(project_root, 'dataset', 'yolo_format'),
                        help='YOLO 格式数据集输出目录')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='模型名称 (yolov8n.pt / yolov8s.pt)')
    args = parser.parse_args()

    # ===== 第一步: 数据预处理 =====
    print("=" * 60)
    print("第一步: 数据预处理 (GTSDB → YOLO 格式)")
    print("=" * 60)

    data_yaml = prepare_dataset(args.dataset, args.output)
    if not os.path.exists(data_yaml):
        print(f"错误: 数据预处理失败，找不到 {data_yaml}")
        sys.exit(1)
    print(f"数据配置文件: {data_yaml}")

    # ===== 第二步: 基准模型训练 =====
    print("\n" + "=" * 60)
    print("第二步: 基准模型训练 (Baseline)")
    print("=" * 60)

    detector = TrafficSignDetector(model_name=args.model)
    baseline_dir = os.path.join(project_root, 'runs', 'baseline')
    if os.path.exists(baseline_dir):
        shutil.rmtree(baseline_dir)

    detector.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device='0',
        project=os.path.join(project_root, 'runs'),
        name='baseline',
        exist_ok=True,
    )

    # 复制最佳权重到结果目录
    best_pt = os.path.join(project_root, 'runs', 'baseline', 'weights', 'best.pt')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    if os.path.exists(best_pt):
        shutil.copy(best_pt, os.path.join(results_dir, 'baseline_best.pt'))
        print(f"基准模型已保存: {os.path.join(results_dir, 'baseline_best.pt')}")

    # ===== 第三步: 改进策略训练 =====
    # 改进策略: 强数据增强 + 余弦退火 LR + 标签平滑
    # 对比 baseline 来验证增强策略是否有效
    print("\n" + "=" * 60)
    print("第三步: 改进策略训练 (强增强 + 余弦LR + 标签平滑)")
    print("=" * 60)

    torch.cuda.empty_cache()

    improved_dir = os.path.join(project_root, 'runs', 'improved')
    if os.path.exists(improved_dir):
        shutil.rmtree(improved_dir)

    detector2 = TrafficSignDetector(model_name=args.model)
    detector2.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device='0',
        project=os.path.join(project_root, 'runs'),
        name='improved',
        exist_ok=True,
        # ===== 改进策略参数 =====
        # 颜色增强 (让模型对光照变化更鲁棒)
        hsv_h=0.05,      # 色调抖动范围
        hsv_s=0.7,       # 饱和度抖动 (更强)
        hsv_v=0.5,       # 明度抖动 (更强)
        # 几何增强 (让模型对视角变化更鲁棒)
        degrees=15.0,    # 旋转 (±15度)
        translate=0.15,  # 平移
        scale=0.6,       # 缩放 (更强的尺度变化)
        shear=5.0,       # 剪切
        perspective=0.0, # 透视变换 (保持关闭)
        flipud=0.1,      # 上下翻转
        fliplr=0.5,      # 左右翻转
        # 高级增强
        mosaic=1.0,       # Mosaic 拼接
        mixup=0.2,        # MixUp 混合
        copy_paste=0.1,   # Copy-paste (小目标增强)
        close_mosaic=15,  # 最后15轮关闭mosaic,稳定精调
        # 学习率调度
        lr0=0.01,         # 初始LR
        lrf=0.001,        # 最终LR
        cos_lr=True,      # 余弦退火调度
        # 正则化
        label_smoothing=0.05,  # 标签平滑,防止过拟合
        warmup_epochs=5.0,     # 预热轮数
        patience=30,           # 早停耐心 (更长)
    )

    improved_pt = os.path.join(project_root, 'runs', 'improved', 'weights', 'best.pt')
    if os.path.exists(improved_pt):
        shutil.copy(improved_pt, os.path.join(results_dir, 'improved_best.pt'))
        print(f"改进模型已保存: {os.path.join(results_dir, 'improved_best.pt')}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"基准模型: {os.path.join(results_dir, 'baseline_best.pt')}")
    print(f"改进模型: {os.path.join(results_dir, 'improved_best.pt')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
