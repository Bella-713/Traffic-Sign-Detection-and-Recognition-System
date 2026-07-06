"""
单独训练 baseline 模型
"""
import os, sys, shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'code'))

from model import TrafficSignDetector


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    torch.cuda.empty_cache()

    data_yaml = os.path.join(project_root, 'dataset', 'yolo_format', 'data.yaml')
    baseline_dir = os.path.join(project_root, 'runs', 'baseline')
    if os.path.exists(baseline_dir):
        shutil.rmtree(baseline_dir)

    detector = TrafficSignDetector(model_name='yolov8s.pt')
    detector.train(
        data_yaml=data_yaml,
        epochs=100,
        batch=8,
        imgsz=640,
        device='0',
        project=os.path.join(project_root, 'runs'),
        name='baseline',
        exist_ok=True,
        patience=15,
    )

    best_pt = os.path.join(project_root, 'runs', 'baseline', 'weights', 'best.pt')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    if os.path.exists(best_pt):
        shutil.copy(best_pt, os.path.join(results_dir, 'baseline_best.pt'))
        print(f'Baseline saved: {os.path.join(results_dir, "baseline_best.pt")}')


if __name__ == '__main__':
    main()
