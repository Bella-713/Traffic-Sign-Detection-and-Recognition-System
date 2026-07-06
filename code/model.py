"""
交通标志检测 - 模型定义
基于 Ultralytics YOLOv8
"""

import os
import shutil
from ultralytics import YOLO


class TrafficSignDetector:
    """交通标志检测模型封装"""

    def __init__(self, model_name='yolov8s.pt'):
        """
        初始化检测器
        Args:
            model_name: 预训练模型名称或权重文件路径
                       可选: 'yolov8n.pt' (nano), 'yolov8s.pt' (small)
                       'yolov8m.pt' (medium), 'yolov8l.pt' (large)
        """
        self.model_name = model_name
        self.model = None

    def train(self, data_yaml, epochs=100, batch=16, imgsz=640, device='0',
              project='runs', name='traffic_sign', exist_ok=False, patience=15,
              **kwargs):
        """
        训练模型
        Args:
            data_yaml: data.yaml 文件路径
            epochs: 训练轮数
            batch: batch size
            imgsz: 输入图像大小
            device: 训练设备 ('cpu', 'cuda:0', 'auto')
            project: 保存项目名
            name: 实验名
            exist_ok: 是否覆盖已有结果
            patience: 早停耐心值
            **kwargs: 传递给 YOLO.train() 的额外参数（增强、LR调度等）
        Returns:
            训练结果对象
        """
        self.model = YOLO(self.model_name)
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            exist_ok=exist_ok,
            patience=patience,
            verbose=True,
            **kwargs,
        )
        return results

    def resume_train(self, weight_path, data_yaml, epochs=50, batch=16, imgsz=640):
        """从断点恢复训练"""
        self.model = YOLO(weight_path)
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            resume=True,
            verbose=True,
        )
        return results

    def evaluate(self, weight_path, data_yaml, batch=16, imgsz=640):
        """
        评估模型
        Args:
            weight_path: 训练好的权重文件 (best.pt)
            data_yaml: data.yaml 路径
        Returns:
            评估结果 (包含 mAP, precision, recall 等)
        """
        self.model = YOLO(weight_path)
        results = self.model.val(
            data=data_yaml,
            batch=batch,
            imgsz=imgsz,
            device='0',
            verbose=True,
        )
        return results

    def predict(self, image_path, weight_path, conf=0.25, iou=0.45):
        """
        单张图片预测
        Args:
            image_path: 图片路径
            weight_path: 模型权重路径
            conf: 置信度阈值
            iou: NMS IoU 阈值
        Returns:
            检测结果
        """
        self.model = YOLO(weight_path)
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        return results

    def export_onnx(self, weight_path, output_path=None):
        """
        导出为 ONNX 格式 (加分项)
        """
        self.model = YOLO(weight_path)
        onnx_path = self.model.export(format='onnx', imgsz=640)
        if output_path and os.path.exists(onnx_path):
            shutil.copy(onnx_path, output_path)
        return onnx_path


def get_best_model(gpu_memory_gb=None):
    """
    根据显存大小推荐模型
    Args:
        gpu_memory_gb: GPU 显存 (GB)
    Returns:
        模型名称字符串
    """
    if gpu_memory_gb is None:
        return 'yolov8s.pt'
    if gpu_memory_gb < 4:
        return 'yolov8n.pt'
    elif gpu_memory_gb < 8:
        return 'yolov8s.pt'
    else:
        return 'yolov8m.pt'
