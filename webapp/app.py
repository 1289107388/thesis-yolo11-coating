# -*- coding: utf-8 -*-
"""毕业设计网页：风机塔筒涂层缺陷检测演示系统（Flask）。

启动：cd 02_代码 && python -m webapp.app   → http://127.0.0.1:8000
推理用 CPU，避免与训练抢 GPU。
"""
import base64
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REPO = ROOT / "CoatingDet-Defect-Detection"
DATA = Path(r"F:\CoatingDet_v2")
sys.path.insert(0, str(ROOT))

app = Flask(__name__)

# 可选模型（启动后可继续扩充 runs 下新的 best.pt）
MODELS = {
    "A0 基线 YOLOv11n": str(REPO / "runs" / "yolo11n_20ep" / "weights" / "best.pt"),
    "C +CBAM 注意力": str(REPO / "runs" / "C_cbam" / "weights" / "best.pt"),
    "本文最终 C@60ep": str(REPO / "runs" / "C_60ep" / "weights" / "best.pt"),
    "低误报变体 F2@60ep": str(REPO / "runs" / "F2_60ep" / "weights" / "best.pt"),
}
_cache = {}


def get_model(path):
    import cbam  # noqa: F401 注册 CBAM 模块，加载含 CBAM 的权重时需要
    import ultralytics.nn.tasks as tasks
    tasks.CBAM = cbam.CBAM
    from ultralytics import YOLO
    if path not in _cache:
        _cache[path] = YOLO(path)
    return _cache[path]


def pick_samples(per_domain=6):
    """每个测试域选标注框最多的若干切片作为演示样例。"""
    out = []
    for domain in ["test_1", "test_2", "test_3"]:
        img_dir = DATA / domain / "tiles" / "images"
        lbl_dir = DATA / domain / "tiles" / "labels"
        scored = []
        for lbl in lbl_dir.glob("*.txt"):
            n = sum(1 for line in lbl.read_text().splitlines() if line.strip())
            scored.append((n, lbl.stem))
        scored.sort(reverse=True)
        for n, stem in scored[:per_domain]:
            img = img_dir / f"{stem}.jpg"
            if img.exists():
                out.append({"domain": domain, "name": stem, "boxes": n})
    return out


SAMPLES = pick_samples()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def models():
    return jsonify({"models": list(MODELS.keys()),
                    "samples": [{"url": f"/sample/{s['domain']}/{s['name']}", **s} for s in SAMPLES]})


@app.get("/sample/<domain>/<name>")
def sample(domain, name):
    if domain not in ("test_1", "test_2", "test_3"):
        return "bad domain", 400
    return send_file(DATA / domain / "tiles" / "images" / f"{name}.jpg")


@app.post("/api/detect")
def detect():
    import cv2
    import numpy as np
    key = request.form.get("model", list(MODELS)[0])
    conf = float(request.form.get("conf", 0.25))
    path = MODELS.get(key)
    if not path:
        return jsonify(error="未知模型"), 400

    if "file" in request.files and request.files["file"].filename:
        buf = np.frombuffer(request.files["file"].read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        ref = request.form.get("sample", "")
        try:
            domain, name = ref.split("/", 1)
            img = cv2.imread(str(DATA / domain / "tiles" / "images" / f"{name}.jpg"))
        except Exception:
            return jsonify(error="请选择样例图片或上传文件"), 400
    if img is None:
        return jsonify(error="图片读取失败"), 400

    r = get_model(path).predict(img, imgsz=512, conf=conf, device="cpu", verbose=False)[0]
    ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 88])
    dets = []
    for b in r.boxes:
        dets.append({
            "cls": r.names[int(b.cls)],
            "conf": round(float(b.conf), 3),
            "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        })
    counts = {}
    for d in dets:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    return jsonify(image="data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                   detections=dets, counts=counts, conf=conf, model=key)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
