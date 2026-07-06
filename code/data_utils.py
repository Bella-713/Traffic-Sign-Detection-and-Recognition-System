"""
交通标志检测 - 数据处理工具
功能: 加载GTSDB、类别映射、格式转换、场景分类、数据增强
"""

import os
import cv2
import shutil
import random
import numpy as np
from glob import glob
from pathlib import Path


# ===== GTSDB 43类 → 作业要求的5类 映射 =====
# 要求类别: 0=限速, 1=禁止通行, 2=直行/转弯, 3=人行横道, 4=停车让行
GTSDB_TO_OURS = {
    # 限速标志 (speed limits)
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 8: 0,
    # 禁止通行 (no entry)
    17: 1,
    # 直行/转弯 (mandatory direction)
    33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2,
    # 人行横道 (pedestrian crossing)
    27: 3,
    # 停车让行 (stop / yield)
    13: 4, 14: 4,
}

OUR_CLASS_NAMES = ['speed_limit', 'no_entry', 'direction', 'crosswalk', 'stop_yield']


def parse_gt_file(gt_path):
    """读取 GTSDB 的 gt.txt 文件"""
    boxes = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) != 6:
                continue
            img_name, x1, y1, x2, y2, cls_id = parts
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            if img_name not in boxes:
                boxes[img_name] = []
            boxes[img_name].append((x1, y1, x2, y2, cls_id))
    return boxes


def convert_to_yolo_format(boxes, img_w, img_h):
    """将角坐标 (x1,y1,x2,y2) 转换为 YOLO 格式 (cx, cy, w, h) 归一化"""
    yolo_boxes = []
    for x1, y1, x2, y2, cls_id in boxes:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        yolo_boxes.append((cls_id, cx, cy, w, h))
    return yolo_boxes


def prepare_dataset(gtsdb_root, output_root, val_split=0.2):
    """
    主函数: 将 GTSDB 数据集转换为 YOLO 格式
    gtsdb_root: GTSDB 数据集根目录 (包含 gt.txt, TestIJCNN2013/, TrainIJCNN2013/)
    output_root: 输出目录
    """
    # 找 gt.txt
    gt_path = os.path.join(gtsdb_root, 'gt.txt')
    if not os.path.exists(gt_path):
        for f in glob(os.path.join(gtsdb_root, '**', 'gt.txt'), recursive=True):
            gt_path = f
            break
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"找不到 gt.txt，请确认数据集路径: {gtsdb_root}")

    # 读取标注
    all_boxes = parse_gt_file(gt_path)

    # 只在 TrainIJCNN2013 中搜索图片（gt.txt 只包含训练集标注）
    # gt.txt 用纯文件名引用图片，我们需要找到对应的实际文件
    train_dir = os.path.join(gtsdb_root, 'TrainIJCNN2013')
    all_images = []
    for ext in ['*.ppm', '*.jpg', '*.png', '*.jpeg']:
        all_images.extend(glob(os.path.join(train_dir, '**', ext), recursive=True))

    # 构建 basename → full path 映射，优先使用非嵌套目录下的文件
    img_path_map = {}
    for p in all_images:
        base = os.path.basename(p)
        if base not in img_path_map:
            img_path_map[base] = p
        else:
            # 如果现有路径在嵌套子目录（如 00/）而新路径不在，用新的
            existing_depth = img_path_map[base].count(os.path.sep)
            new_depth = p.count(os.path.sep)
            if new_depth < existing_depth:
                img_path_map[base] = p

    print(f"找到 {len(img_path_map)} 张训练图片（去重后）")

    img_names = list(img_path_map.keys())

    # 只保留有标注且包含我们所需类别的图片
    valid_images = []
    for img_name in img_names:
        if img_name not in all_boxes:
            continue
        filtered = [b for b in all_boxes[img_name] if b[4] in GTSDB_TO_OURS]
        if len(filtered) > 0:
            valid_images.append((img_name, filtered))

    print(f"有效图片数: {len(valid_images)} / {len(img_names)}")

    # 随机打乱并划分训练/验证集（固定种子保证可复现）
    random.seed(42)
    random.shuffle(valid_images)
    val_count = int(len(valid_images) * val_split)
    val_images = valid_images[:val_count]
    train_images = valid_images[val_count:]

    print(f"训练集: {len(train_images)}, 验证集: {len(val_images)}")

    # 创建输出目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'labels', split), exist_ok=True)

    # 处理并写入
    def process_split(image_list, split_name):
        for img_name, boxes in image_list:
            src_path = img_path_map[img_name]

            # 读取图片获取尺寸
            img = cv2.imread(src_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            # 复制图片 (转jpg)
            dst_img_name = os.path.splitext(img_name)[0] + '.jpg'
            dst_img_path = os.path.join(output_root, 'images', split_name, dst_img_name)
            cv2.imwrite(dst_img_path, img)

            # 转换标注并写入
            mapped_boxes = []
            for x1, y1, x2, y2, cls_id in boxes:
                new_cls = GTSDB_TO_OURS[cls_id]
                mapped_boxes.append((x1, y1, x2, y2, new_cls))

            yolo_boxes = convert_to_yolo_format(mapped_boxes, img_w, img_h)
            label_path = os.path.join(output_root, 'labels', split_name,
                                      os.path.splitext(img_name)[0] + '.txt')
            with open(label_path, 'w') as f:
                for cls_id, cx, cy, w, h in yolo_boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    process_split(train_images, 'train')
    process_split(val_images, 'val')

    # 生成 data.yaml
    data_yaml = f"""train: {os.path.join(output_root, 'images', 'train')}
val: {os.path.join(output_root, 'images', 'val')}
nc: 5
names: ['speed_limit', 'no_entry', 'direction', 'crosswalk', 'stop_yield']
"""
    yaml_path = os.path.join(output_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)

    print(f"数据集准备完成! 配置文件: {yaml_path}")
    return yaml_path


def classify_scene(image_path):
    """
    根据图像亮度和模糊度分类场景
    返回: 'bright' / 'normal' / 'dark' / 'blurry'
    """
    img = cv2.imread(image_path)
    if img is None:
        return 'normal'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 亮度: 计算灰度均值
    brightness = np.mean(gray)

    # 模糊度: 拉普拉斯方差 (值越低越模糊)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if brightness < 80:
        return 'dark'
    elif laplacian_var < 100:
        return 'blurry'
    elif brightness > 180:
        return 'bright'
    else:
        return 'normal'


def get_augmentation_pipeline():
    """
    返回改进策略使用的增强流程 (用于对比 baseline)
    包括: 低光模拟、运动模糊、高斯噪声
    """
    import albumentations as A
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])


if __name__ == '__main__':
    # 如果直接运行此脚本，进行数据预处理
    import sys
    if len(sys.argv) > 1:
        gtsdb_path = sys.argv[1]
    else:
        gtsdb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'yolo_format')
    prepare_dataset(gtsdb_path, output_path)
    print("数据预处理完成!")
