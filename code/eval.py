"""
交通标志检测 - 评估脚本
生成: 混淆矩阵、损失曲线、场景对比分析表、改进前后对比
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'code'))

from data_utils import classify_scene, OUR_CLASS_NAMES
from model import TrafficSignDetector


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results_dict(results_obj):
    """从 YOLO 评估结果中提取指标"""
    def to_scalar(v):
        """将 tensor/array 转为标量"""
        if hasattr(v, 'cpu'):
            return v.cpu().mean().item()
        if hasattr(v, 'mean'):
            return float(v.mean())
        return float(v) if v else 0

    metrics = {
        'mAP50': to_scalar(results_obj.box.map50) if hasattr(results_obj, 'box') else 0,
        'mAP50_95': to_scalar(results_obj.box.map) if hasattr(results_obj, 'box') else 0,
        'precision': to_scalar(results_obj.box.p) if hasattr(results_obj, 'box') else 0,
        'recall': to_scalar(results_obj.box.r) if hasattr(results_obj, 'box') else 0,
        'f1': 0,
    }
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / \
                         (metrics['precision'] + metrics['recall'])
    return metrics


def plot_loss_curve(train_dir, save_path):
    """
    从训练结果中读取损失曲线并绘制
    train_dir: runs/baseline 或 runs/improved 目录
    save_path: 保存路径
    """
    results_csv = os.path.join(train_dir, 'results.csv')
    if not os.path.exists(results_csv):
        print(f"警告: 找不到训练结果文件 {results_csv}")
        return

    import pandas as pd
    df = pd.read_csv(results_csv)

    # 去除列名中的空格
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    if loss_cols:
        ax = axes[0]
        for col in loss_cols[:3]:  # 最多画3条损失曲线
            short_name = col.replace('/train', '').replace('/val', '').strip()
            ax.plot(df.index, df[col], label=short_name, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training / Validation Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 精度曲线
    prec_cols = [c for c in df.columns if 'precision' in c.lower() or 'mAP' in c.lower()]
    if prec_cols:
        ax = axes[1]
        for col in prec_cols:
            ax.plot(df.index, df[col], label=col.strip(), linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.set_title('Precision')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # 尝试 mAP 相关列
        map_cols = [c for c in df.columns if 'map' in c.lower()]
        if map_cols:
            ax = axes[1]
            for col in map_cols:
                ax.plot(df.index, df[col], label=col.strip(), linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('mAP')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    # 召回率曲线
    recall_cols = [c for c in df.columns if 'recall' in c.lower() or 'recall' in c.lower()]
    if recall_cols:
        ax = axes[2]
        for col in recall_cols:
            ax.plot(df.index, df[col], label=col.strip(), linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.set_title('Recall')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No recall data', ha='center', va='center')
        axes[2].set_title('Recall')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存: {save_path}")


def plot_confusion_matrix(weight_path, data_yaml, save_path):
    """
    使用 YOLO 验证结果生成混淆矩阵
    优先利用 ultralytics 生成的混淆矩阵,若找不到则通过 model.val() 实时计算
    """
    # 尝试从 ultralytics 验证结果中找已生成的混淆矩阵
    runs_dir = os.path.join(project_root, 'runs')
    conf_matrix_path = None
    for cm_path in glob(os.path.join(runs_dir, '**', 'confusion_matrix.png'), recursive=True):
        conf_matrix_path = cm_path
        break

    if conf_matrix_path and os.path.exists(conf_matrix_path):
        # 复制已生成的混淆矩阵
        import shutil
        shutil.copy(conf_matrix_path, save_path)
        print(f"混淆矩阵已复制: {save_path}")
        return

    # 未找到已生成的矩阵,运行 val 获取真实混淆矩阵数据
    print("未找到已有混淆矩阵,正在通过验证实时计算...")
    from ultralytics import YOLO
    model = YOLO(weight_path)
    val_results = model.val(data=data_yaml, verbose=False, save_json=False)

    if hasattr(val_results, 'confusion_matrix') and val_results.confusion_matrix is not None:
        cm_matrix = val_results.confusion_matrix.matrix
    else:
        print("警告: 无法获取混淆矩阵,跳过")
        return

    # 截取前 nc 行/列（去掉背景类）
    with open(data_yaml) as f:
        import yaml
        nc = yaml.safe_load(f).get('nc', 5)
    cm_matrix = cm_matrix[:nc, :nc]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=OUR_CLASS_NAMES, yticklabels=OUR_CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def analyze_scene_conditions(data_yaml, weight_path, save_path):
    """
    按场景条件分析性能（使用真实 mAP 指标）
    将验证集图片按亮度/模糊度分组，对每组运行 YOLO val 计算真实 mAP
    """
    import yaml
    import cv2
    import random
    import shutil
    from ultralytics import YOLO

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    val_img_dir = data_cfg.get('val', '')
    val_label_dir = val_img_dir.replace('images', 'labels')
    if not val_img_dir or not os.path.exists(val_img_dir):
        print(f"警告: 找不到验证集图片目录 {val_img_dir}")
        return

    model = YOLO(weight_path)

    img_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        img_files.extend(glob(os.path.join(val_img_dir, ext)))
    if not img_files:
        print("验证集没有图片")
        return

    print(f"场景分析: 共 {len(img_files)} 张验证图片")

    # 按场景分类
    scene_groups = {'bright': [], 'normal': [], 'dark': [], 'blurry': []}
    for img_path in img_files:
        scene = classify_scene(img_path)
        scene_groups[scene].append(img_path)

    results_summary = {}
    yolo_root = os.path.dirname(os.path.dirname(data_yaml))

    for scene, imgs in scene_groups.items():
        if len(imgs) == 0:
            continue

        # 随机采样（最多 50 张）
        sample_imgs = random.sample(imgs, min(50, len(imgs)))

        # 创建临时场景数据集
        temp_dir = os.path.join(yolo_root, f'_scene_{scene}')
        temp_img_dir = os.path.join(temp_dir, 'images')
        temp_label_dir = os.path.join(temp_dir, 'labels')
        os.makedirs(temp_img_dir, exist_ok=True)
        os.makedirs(temp_label_dir, exist_ok=True)

        for img_path in sample_imgs:
            shutil.copy(img_path, os.path.join(temp_img_dir, os.path.basename(img_path)))
            lbl_src = os.path.join(
                val_label_dir,
                os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(temp_label_dir, os.path.basename(lbl_src)))

        temp_yaml = os.path.join(temp_dir, 'data.yaml')
        with open(temp_yaml, 'w') as f:
            f.write(f"train: {temp_img_dir}\nval: {temp_img_dir}\nnc: 5\nnames: {OUR_CLASS_NAMES}\n")

        try:
            val_results = model.val(data=temp_yaml, verbose=False, save_json=False)

            def to_scalar(v):
                if hasattr(v, 'cpu'):
                    return v.cpu().mean().item()
                if hasattr(v, 'mean'):
                    return float(v.mean())
                return float(v) if v else 0

            results_summary[scene] = {
                'count': len(imgs),
                'sampled': len(sample_imgs),
                'mAP50': round(to_scalar(val_results.box.map50), 3),
                'mAP50_95': round(to_scalar(val_results.box.map), 3),
                'precision': round(to_scalar(val_results.box.p), 3),
                'recall': round(to_scalar(val_results.box.r), 3),
            }
            print(f"  场景 {scene:8s}: mAP@0.5={results_summary[scene]['mAP50']:.3f}, "
                  f"采样 {len(sample_imgs)} 张")
        except Exception as e:
            print(f"  场景 {scene} 评估失败: {e}")
            results_summary[scene] = {
                'count': len(imgs), 'sampled': len(sample_imgs),
                'mAP50': 0, 'mAP50_95': 0, 'precision': 0, 'recall': 0,
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # 绘制对比表格
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis('off')

    col_labels = ['场景', '图片数', '采样数', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    scene_names = {'bright': '明亮', 'normal': '正常', 'dark': '低光(夜间)', 'blurry': '模糊'}
    table_data = []
    for scene in ['bright', 'normal', 'dark', 'blurry']:
        info = results_summary.get(scene)
        if info:
            table_data.append([
                scene_names.get(scene, scene), info['count'], info['sampled'],
                info['mAP50'], info['mAP50_95'], info['precision'], info['recall'],
            ])

    if table_data:
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('不同场景条件下的检测性能对比 (mAP@0.5)', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"场景对比表已保存: {save_path}")
    return results_summary


def compute_iou(box1, box2):
    """计算两个矩形框的 IoU（xyxy 格式）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def analyze_false_detections(weight_path, data_yaml, save_dir, iou_threshold=0.5, max_samples=100):
    """
    分析模型的误检(FP)和漏检(FN)案例,生成可视化图表与统计数据
    """
    import yaml
    from ultralytics import YOLO
    import cv2
    import random

    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    val_img_dir = data_cfg['val']
    label_dir = val_img_dir.replace('images', 'labels')
    if not os.path.exists(val_img_dir):
        print(f"跳过误检分析: 找不到 {val_img_dir}")
        return None, 0

    print(f"加载模型: {weight_path}")
    model = YOLO(weight_path)

    img_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        img_files.extend(glob(os.path.join(val_img_dir, ext)))
    if not img_files:
        print("跳过误检分析: 无验证图片")
        return None, 0

    sample_size = min(max_samples, len(img_files))
    sampled = random.sample(img_files, sample_size)

    stats = {'TP': 0, 'FP': 0, 'FN': 0, 'total_gt': 0}
    fp_cases, fn_cases, tp_cases = [], [], []

    for img_path in sampled:
        # 读取 GT
        label_path = os.path.join(
            label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        gt_yolo = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_yolo.append(tuple(map(float, parts)))

        # 预测
        results = model.predict(img_path, conf=0.25, verbose=False)
        pred_list = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                pred_list.append((int(boxes.cls[i].item()), *xyxy, boxes.conf[i].item()))

        if not gt_yolo and not pred_list:
            continue

        stats['total_gt'] += len(gt_yolo)

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        # GT → xyxy
        gt_xyxy = []
        for cls_id, cx, cy, w, h in gt_yolo:
            x1 = (cx - w / 2) * w_img
            y1 = (cy - h / 2) * h_img
            x2 = (cx + w / 2) * w_img
            y2 = (cy + h / 2) * h_img
            gt_xyxy.append((int(cls_id), x1, y1, x2, y2))

        # IoU 匹配
        matched_gt, matched_pred = set(), set()
        for pi, (p_cls, px1, py1, px2, py2, p_conf) in enumerate(pred_list):
            best_iou, best_gi = 0, -1
            for gi, (g_cls, gx1, gy1, gx2, gy2) in enumerate(gt_xyxy):
                if gi in matched_gt:
                    continue
                iou = compute_iou([px1, py1, px2, py2], [gx1, gy1, gx2, gy2])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                stats['TP'] += 1
                if len(tp_cases) < 3:
                    tp_cases.append((img_path, [pred_list[pi]], [gt_xyxy[best_gi]]))
            else:
                stats['FP'] += 1
                if len(fp_cases) < 5:
                    fp_cases.append((img_path, [pred_list[pi]], gt_xyxy))

        for gi in range(len(gt_xyxy)):
            if gi not in matched_gt:
                stats['FN'] += 1
                if len(fn_cases) < 5:
                    fn_cases.append((img_path, [], [gt_xyxy[gi]]))
                    break  # 每张图最多一个 FN，避免单一图片占满

    precision = stats['TP'] / max(stats['TP'] + stats['FP'], 1)
    recall = stats['TP'] / max(stats['total_gt'], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    print(f"\n误检/漏检分析 (采样 {sample_size} 张):")
    print(f"  TP={stats['TP']}  FP={stats['FP']}  FN={stats['FN']}  "
          f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    # ===== 绘制可视化 =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    plot_idx = 0

    def draw_case(ax, img_path, preds, gts, title):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            ax.text(0.5, 0.5, 'Load error', ha='center', va='center')
            ax.axis('off')
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # GT (绿)
        for gt in gts:
            cls_id, x1, y1, x2, y2 = gt[:5]
            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(img_rgb, f"GT:{OUR_CLASS_NAMES[int(cls_id)]}",
                        (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        # Pred (蓝)
        for p in preds:
            cls_id, x1, y1, x2, y2, conf = p
            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (200, 0, 0), 2)
            cv2.putText(img_rgb, f"Pred:{OUR_CLASS_NAMES[int(cls_id)]} {conf:.2f}",
                        (int(x1), int(y2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # TP / FP / FN 各放 1~2 张
    for label, cases, color_prefix in [('TP', tp_cases[:2], 'TP Correct'),
                                        ('FP', fp_cases[:2], 'FP False Positive'),
                                        ('FN', fn_cases[:2], 'FN False Negative')]:
        for i, (img_path, preds, gts) in enumerate(cases):
            if plot_idx < 6:
                draw_case(axes_flat[plot_idx], img_path, preds, gts,
                          f'{color_prefix} #{i + 1}')
                plot_idx += 1

    # 统计面板
    if plot_idx < 6:
        ax = axes_flat[plot_idx]
        ax.axis('off')
        text = (
            f"误检/漏检统计 (n={sample_size})\n\n"
            f"TP (正确检测):  {stats['TP']}\n"
            f"FP (误检):     {stats['FP']}\n"
            f"FN (漏检):     {stats['FN']}\n\n"
            f"Precision: {precision:.1%}\n"
            f"Recall:    {recall:.1%}\n"
            f"F1-score:  {f1:.1%}"
        )
        ax.text(0.1, 0.9, text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

    for i in range(plot_idx + 1, 6):
        axes_flat[i].axis('off')

    plt.suptitle('误检 / 漏检案例分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'false_detection_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"误检分析图已保存: {save_path}")

    return stats, sample_size


def compare_baseline_vs_improved():
    """对比基准模型和改进模型的性能"""
    results_dir = os.path.join(project_root, 'results')
    baseline_pt = os.path.join(results_dir, 'baseline_best.pt')
    improved_pt = os.path.join(results_dir, 'improved_best.pt')
    data_yaml = os.path.join(project_root, 'dataset', 'yolo_format', 'data.yaml')

    comparison = {}
    detector = TrafficSignDetector()

    if os.path.exists(baseline_pt) and os.path.exists(data_yaml):
        print("\n评估基准模型...")
        results = detector.evaluate(baseline_pt, data_yaml)
        comparison['baseline'] = load_results_dict(results)

    if os.path.exists(improved_pt) and os.path.exists(data_yaml):
        print("\n评估改进模型...")
        results = detector.evaluate(improved_pt, data_yaml)
        comparison['improved'] = load_results_dict(results)

    return comparison, data_yaml


def main():
    print("=" * 60)
    print("交通标志检测 - 模型评估")
    print("=" * 60)

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ===== 1. 对比基准 vs 改进 =====
    print("\n[1/6] 评估模型性能...")
    comparison, data_yaml = compare_baseline_vs_improved()

    if comparison:
        print("\n" + "-" * 40)
        print(f"{'指标':<15} {'基准模型':<12} {'改进模型':<12}")
        print("-" * 40)
        for metric in ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']:
            base = comparison.get('baseline', {}).get(metric, 0)
            impr = comparison.get('improved', {}).get(metric, 0)
            base_str = f"{base:.3f}" if base else "N/A"
            impr_str = f"{impr:.3f}" if impr else "N/A"
            print(f"{metric:<15} {base_str:<12} {impr_str:<12}")
        print("-" * 40)

        # 保存比较结果到 JSON
        with open(os.path.join(results_dir, 'comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
        print("对比结果已保存")

    # ===== 2. 生成损失曲线 =====
    print("\n[2/6] 生成损失曲线...")
    for model_type in ['baseline', 'improved']:
        train_dir = os.path.join(project_root, 'runs', model_type)
        if os.path.exists(train_dir):
            save_path = os.path.join(results_dir, f'loss_curve_{model_type}.png')
            plot_loss_curve(train_dir, save_path)

    # 如果只有一份，复制一份作为总的 loss_curve.png
    if os.path.exists(os.path.join(results_dir, 'loss_curve_baseline.png')):
        import shutil
        shutil.copy(os.path.join(results_dir, 'loss_curve_baseline.png'),
                    os.path.join(results_dir, 'loss_curve.png'))

    # ===== 3. 生成混淆矩阵 =====
    print("\n[3/6] 生成混淆矩阵...")
    baseline_pt = os.path.join(results_dir, 'baseline_best.pt')
    improved_pt = os.path.join(results_dir, 'improved_best.pt')

    if os.path.exists(baseline_pt) and data_yaml:
        plot_confusion_matrix(baseline_pt, data_yaml,
                              os.path.join(results_dir, 'confusion_matrix.png'))

    # ===== 4. 场景对比分析 =====
    print("\n[4/6] 场景对比分析...")
    if os.path.exists(baseline_pt) and data_yaml:
        analyze_scene_conditions(
            data_yaml, baseline_pt,
            os.path.join(results_dir, 'comparison_table.png')
        )

    # ===== 5. 误检/漏检分析 =====
    print("\n[5/6] 误检/漏检分析...")
    if os.path.exists(baseline_pt) and data_yaml:
        analyze_false_detections(
            baseline_pt, data_yaml, results_dir, max_samples=100
        )

    # ===== 6. 汇总报告 =====
    print("\n[6/6] 生成评估报告...")
    report_path = os.path.join(results_dir, 'eval_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("交通标志检测系统 - 评估报告\n")
        f.write("=" * 50 + "\n\n")

        for model_type in ['baseline', 'improved']:
            if model_type in comparison:
                f.write(f"\n{model_type.upper()} 模型:\n")
                f.write("-" * 30 + "\n")
                for metric, value in comparison[model_type].items():
                    f.write(f"  {metric}: {value:.4f}\n")

        f.write("\n\n生成的文件:\n")
        f.write(f"  - {os.path.join(results_dir, 'loss_curve.png')}\n")
        f.write(f"  - {os.path.join(results_dir, 'confusion_matrix.png')}\n")
        f.write(f"  - {os.path.join(results_dir, 'comparison_table.png')}\n")
        f.write(f"  - {os.path.join(results_dir, 'false_detection_analysis.png')}\n")

    print(f"评估报告已保存: {report_path}")
    print("\n评估完成! 所有结果文件在 results/ 目录下。")


if __name__ == '__main__':
    main()
