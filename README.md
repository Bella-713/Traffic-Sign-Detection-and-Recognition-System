# Traffic Sign Detection System Based on Improved YOLOv8
This project implements traffic sign detection using YOLOv8 on the GTSDB dataset. We convert the original 43 categories into 5 practical driving classes and adopt multiple optimization strategies to improve model robustness.

## Project Intro
- Dataset: GTSDB, 900 road images with traffic sign annotations
- Baseline: YOLOv8s; Optimizations: strong data augmentation, cosine annealing LR, label smoothing, MixUp & Copy-Paste
- Final Result: mAP@0.5 reaches 0.9015, recall increased by 9.9%, effectively reducing missed detections
- Built a web demo for image detection visualization

## File Structure
