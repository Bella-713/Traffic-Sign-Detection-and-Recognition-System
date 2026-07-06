"""
交通标志检测 - Web 演示
在浏览器中上传图片，实时查看检测结果
启动: python code/web_demo.py
访问: http://localhost:5000
"""
import os, sys, io, base64
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'code'))

from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from data_utils import OUR_CLASS_NAMES

app = Flask(__name__)

CLASS_CN = {
    'speed_limit': '限速',
    'no_entry': '禁止通行',
    'direction': '直行/转弯',
    'crosswalk': '人行横道',
    'stop_yield': '停车让行',
}

COLORS = [
    (0, 255, 0),    # 绿 - speed_limit
    (0, 0, 255),    # 红 - no_entry
    (255, 0, 0),    # 蓝 - direction
    (255, 255, 0),  # 青 - crosswalk
    (0, 255, 255),  # 黄 - stop_yield
]

# 加载模型
MODEL_PATH = None
for candidate in [
    os.path.join(project_root, 'results', 'improved_best.pt'),
    os.path.join(project_root, 'results', 'baseline_best.pt'),
]:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

model = None
if MODEL_PATH:
    model = YOLO(MODEL_PATH)
    print(f'模型已加载: {MODEL_PATH}')
else:
    print('警告: 未找到模型权重，请先训练')

HTML = r"""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>交通标志检测 Demo</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: linear-gradient(135deg, #1a1a2e, #16213e); min-height: 100vh; color: #eee; }
.header { text-align: center; padding: 30px 20px 10px; }
.header h1 { font-size: 28px; background: linear-gradient(90deg, #00c6ff, #0072ff);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header p { color: #888; margin-top: 6px; }
.container { max-width: 1000px; margin: 0 auto; padding: 20px; display: flex; gap: 20px; flex-wrap: wrap; }
.panel { flex: 1; min-width: 300px; background: rgba(255,255,255,0.05);
         border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.1); }
.panel h2 { font-size: 18px; margin-bottom: 16px; color: #00c6ff; }
.upload-area { border: 2px dashed rgba(255,255,255,0.2); border-radius: 12px;
               padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.3s; }
.upload-area:hover { border-color: #00c6ff; background: rgba(0,198,255,0.05); }
.upload-area.dragover { border-color: #00c6ff; background: rgba(0,198,255,0.1); }
.upload-area input { display: none; }
.upload-icon { font-size: 40px; margin-bottom: 10px; }
.upload-hint { color: #888; font-size: 14px; }
#result-img { width: 100%; border-radius: 8px; display: none; }
#placeholder { color: #555; text-align: center; padding: 60px 20px; font-size: 14px; }
.result-table { width: 100%; border-collapse: collapse; margin-top: 16px; display: none; }
.result-table th, .result-table td { padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
.result-table th { color: #888; font-size: 12px; text-transform: uppercase; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.conf-high { background: rgba(0,200,0,0.2); color: #0f0; }
.conf-mid { background: rgba(255,200,0,0.2); color: #ff0; }
.summary { margin-top: 20px; display: none; }
.summary-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.loading { text-align: center; padding: 40px; display: none; }
.loading::after { content: ''; display: inline-block; width: 32px; height: 32px;
                  border: 3px solid rgba(255,255,255,0.2); border-top-color: #00c6ff;
                  border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.error { color: #f44; text-align: center; padding: 20px; display: none; }
</style>
</head>
<body>
<div class="header">
  <h1>交通标志检测系统</h1>
  <p>基于 YOLOv8 · GTSDB 数据集 · 5 类交通标志识别</p>
</div>

<div class="container">
  <div class="panel">
    <h2>上传图片</h2>
    <div class="upload-area" id="upload-area">
      <div class="upload-icon">📁</div>
      <div class="upload-hint">点击上传或拖拽图片到此处<br>支持 JPG / PNG</div>
      <input type="file" id="file-input" accept="image/*">
    </div>
    <div class="loading" id="loading"></div>
    <div class="error" id="error"></div>
    <div class="summary" id="summary"></div>
    <table class="result-table" id="result-table">
      <thead><tr><th>类别</th><th>中文</th><th>置信度</th></tr></thead>
      <tbody id="result-tbody"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>检测结果</h2>
    <div id="placeholder">👈 上传一张图片查看检测效果</div>
    <img id="result-img" alt="检测结果">
  </div>
</div>

<script>
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const resultImg = document.getElementById('result-img');
const placeholder = document.getElementById('placeholder');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const summary = document.getElementById('summary');
const table = document.getElementById('result-table');
const tbody = document.getElementById('result-tbody');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault(); uploadArea.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  error.style.display = 'none';
  loading.style.display = 'block';
  resultImg.style.display = 'none';
  placeholder.style.display = 'none';
  table.style.display = 'none';
  summary.style.display = 'none';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const resp = await fetch('/detect', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) { showError(data.error); return; }
    resultImg.src = 'data:image/jpeg;base64,' + data.image;
    resultImg.style.display = 'block';
    loading.style.display = 'none';

    tbody.innerHTML = '';
    data.detections.forEach((d, i) => {
      const tr = document.createElement('tr');
      const confClass = d.confidence >= 0.7 ? 'conf-high' : 'conf-mid';
      tr.innerHTML = `<td><span class="badge" style="background:rgba(${[d.color].join(',')},0.2);color:rgb(${[d.color].join(',')})">${d.class_name}</span></td>
                      <td>${d.class_cn}</td>
                      <td><span class="badge ${confClass}">${(d.confidence*100).toFixed(1)}%</span></td>`;
      tbody.appendChild(tr);
    });
    table.style.display = data.detections.length ? 'table' : 'none';

    summary.innerHTML = `
      <div class="summary-item"><span>检测到标志数</span><span><strong>${data.summary.total}</strong></span></div>
      <div class="summary-item"><span>推理时间</span><span>${data.summary.inference_time}</span></div>
      <div class="summary-item"><span>模型</span><span>${data.summary.model}</span></div>`;
    summary.style.display = 'block';
  } catch(e) {
    showError('网络错误: ' + e.message);
  }
}

function showError(msg) {
  loading.style.display = 'none';
  error.textContent = msg;
  error.style.display = 'block';
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/detect', methods=['POST'])
def detect():
    if model is None:
        return jsonify({'error': '模型未加载，请先完成训练'})

    if 'image' not in request.files:
        return jsonify({'error': '请上传图片'})

    file = request.files['image']
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({'error': '无法解析图片'})

    # 推理
    import time
    t0 = time.time()
    results = model.predict(img_bgr, conf=0.25, iou=0.45, verbose=False)
    t1 = time.time()

    # 用 PIL 画框和中文标签（cv2.putText 不支持中文）
    from PIL import ImageDraw, ImageFont
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 尝试加载中文字体
    font = None
    for font_path in [
        'C:/Windows/Fonts/msyh.ttc',   # 微软雅黑
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 18)
                break
            except Exception:
                continue

    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            cls_id = int(boxes.cls[i])
            conf = boxes.conf[i].item()
            color = COLORS[cls_id]  # BGR -> RGB for PIL
            color_rgb = (color[2], color[1], color[0])

            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)
            # 画中文标签
            label = f'{CLASS_CN.get(OUR_CLASS_NAMES[cls_id], OUR_CLASS_NAMES[cls_id])} {conf:.2f}'
            # 标签背景
            bbox = draw.textbbox((x1, y1 - 22), label, font=font)
            draw.rectangle(bbox, fill=color_rgb)
            draw.text((x1, y1 - 22), label, fill=(255, 255, 255), font=font)

            detections.append({
                'class_name': OUR_CLASS_NAMES[cls_id],
                'class_cn': CLASS_CN.get(OUR_CLASS_NAMES[cls_id], '?'),
                'confidence': round(conf, 4),
                'bbox': [x1, y1, x2, y2],
                'color': list(color),
            })

    # 编码结果图
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({
        'image': img_b64,
        'detections': detections,
        'summary': {
            'total': len(detections),
            'inference_time': f'{(t1 - t0) * 1000:.0f} ms',
            'model': os.path.basename(MODEL_PATH) if MODEL_PATH else 'unknown',
        }
    })


if __name__ == '__main__':
    print('=' * 50)
    print('交通标志检测 Web Demo')
    print(f'模型: {MODEL_PATH}')
    print('访问 http://localhost:5000')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
