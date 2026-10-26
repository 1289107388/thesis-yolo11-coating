# -*- coding: utf-8 -*-
"""毕业设计演示系统 v2：工业检测终端。

启动：cd 02_代码 && python -m webapp.app   → http://127.0.0.1:8000
推理走 CPU，不与训练抢 GPU。
"""
import base64
import csv
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REPO = ROOT / "CoatingDet-Defect-Detection"
DATA = Path(r"F:\CoatingDet_v2")
sys.path.insert(0, str(ROOT))

app = Flask(__name__)

MODELS = {
    "A0 基线": str(REPO / "runs" / "yolo11n_20ep" / "weights" / "best.pt"),
    "C +CBAM（本文）": str(REPO / "runs" / "C_cbam" / "weights" / "best.pt"),
    "E +白平衡": str(REPO / "runs" / "E_gw" / "weights" / "best.pt"),
    "D2 +WIoU": str(REPO / "runs" / "D2_wiou_rclamp" / "weights" / "best.pt"),
}
_cache = {}

# 各域切片索引（启动时构建：文件名 + GT 框）
DOMAINS = {
    "train": "train/tiles/train",
    "val": "train/tiles/val",
    "test_1": "test_1/tiles",
    "test_2": "test_2/tiles",
    "test_3": "test_3/tiles",
}
INDEX = {}
for dom, sub in DOMAINS.items():
    items = []
    for img in sorted((DATA / sub / "images").glob("*.jpg")):
        lbl = DATA / sub / "labels" / f"{img.stem}.txt"
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    boxes.append([int(p[0])] + [round(float(v), 5) for v in p[1:]])
        items.append({"name": img.stem, "boxes": boxes})
    INDEX[dom] = items


def get_model(path):
    import cbam  # noqa: F401
    import ultralytics.nn.tasks as tasks
    tasks.CBAM = cbam.CBAM
    from ultralytics import YOLO
    if path not in _cache:
        _cache[path] = YOLO(path)
    return _cache[path]


def _predict(path, img, conf):
    import cv2
    r = get_model(path).predict(img, imgsz=512, conf=conf, device="cpu", verbose=False)[0]
    ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 88])
    dets = [{"cls": r.names[int(b.cls)], "conf": round(float(b.conf), 3),
             "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()]} for b in r.boxes]
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode(), dets


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def meta():
    return jsonify(
        models=list(MODELS.keys()),
        domains={d: {"tiles": len(v), "boxes": sum(len(x["boxes"]) for x in v)} for d, v in INDEX.items()},
        dataset={"images": 5416, "tiles": 2385, "boxes": 2526, "classes": ["defect", "particle"]},
    )


@app.get("/api/tiles/<domain>")
def tiles(domain):
    if domain not in INDEX:
        return jsonify(error="unknown domain"), 400
    page = max(1, int(request.args.get("page", 1)))
    per = 12
    items = INDEX[domain]
    chunk = items[(page - 1) * per: page * per]
    return jsonify(page=page, pages=(len(items) + per - 1) // per, total=len(items),
                   tiles=[{"url": f"/tile/{domain}/{t['name']}", "name": t["name"], "n": len(t["boxes"])} for t in chunk])


@app.get("/tile/<domain>/<name>")
def tile_file(domain, name):
    if domain not in DOMAINS:
        return "bad domain", 400
    return send_file(DATA / DOMAINS[domain] / "images" / f"{name}.jpg")


@app.get("/api/gt/<domain>/<name>")
def gt(domain, name):
    if domain not in INDEX:
        return jsonify(error="unknown domain"), 400
    for t in INDEX[domain]:
        if t["name"] == name:
            return jsonify(name=name, boxes=t["boxes"], classes=["defect", "particle"])
    return jsonify(error="not found"), 404


@app.get("/api/curves")
def curves():
    out = {}
    for run in ["yolo11n_20ep", "C_cbam", "E_gw", "D2_wiou_rclamp"]:
        f = REPO / "runs" / run / "results.csv"
        if not f.exists():
            continue
        ep, box, m50 = [], [], []
        for row in csv.DictReader(open(f)):
            try:
                ep.append(int(row["epoch"]) + 1)
                box.append(float(row["train/box_loss"]))
                m50.append(float(row["metrics/mAP50(B)"]))
            except (KeyError, ValueError):
                continue
        out[run] = {"epoch": ep, "box_loss": box, "mAP50": m50}
    return jsonify(out)


@app.post("/api/detect")
def detect():
    import cv2
    import numpy as np
    key = request.form.get("model", list(MODELS)[0])
    key2 = request.form.get("model2", "")
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
            img = cv2.imread(str(DATA / DOMAINS[domain] / "images" / f"{name}.jpg"))
        except Exception:
            return jsonify(error="请选择样例或上传图片"), 400
    if img is None:
        return jsonify(error="图片读取失败"), 400
    t0 = time.perf_counter()
    image, dets = _predict(path, img, conf)
    ms = round((time.perf_counter() - t0) * 1000)
    counts = {}
    for d in dets:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    out = {"image": image, "detections": dets, "counts": counts, "conf": conf, "model": key, "ms": ms}
    if key2 and key2 in MODELS and key2 != key:
        image2, dets2 = _predict(MODELS[key2], img, conf)
        counts2 = {}
        for d in dets2:
            counts2[d["cls"]] = counts2.get(d["cls"], 0) + 1
        out.update({"image2": image2, "detections2": dets2, "counts2": counts2, "model2": key2})
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
