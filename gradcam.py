# -*- coding: utf-8 -*-
"""YOLOv11 检测的 Grad-CAM（网页与论文图共用）。

原理：选定特征层（默认骨干末端 C2PSA；CBAM 模型可选 head 里的 CBAM），
取该类最大置信度分数反传，特征图梯度做全局平均得到通道权重，
加权和过 ReLU 得到类激活图，归一化后以热力图叠回原图。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbam  # noqa: F401
import ultralytics.nn.tasks as tasks
tasks.CBAM = cbam.CBAM
from ultralytics import YOLO  # noqa: E402

LAYERS = {"c2psa": "C2PSA", "cbam": "CBAM"}


def find_layer(net, layer, first=False):
    """first=True 取第一个匹配层（CBAM 用 P3 浅层，热力图空间细节更好且非空）。"""
    cls_name = LAYERS.get(layer, layer)
    idxs = [i for i, m in enumerate(net.model) if type(m).__name__ == cls_name]
    if not idxs:
        return None
    return net.model[idxs[0] if first else idxs[-1]]


def available_layers(weights):
    net = YOLO(weights).model
    return {k: find_layer(net, k) is not None for k in LAYERS}


def gradcam(weights, img_bgr, cls=0, layer="c2psa", imgsz=512, alpha=0.45):
    """返回 (热力叠加图 BGR, 激活图 float32 HxW, info)。cls: 0=defect 1=particle。"""
    model = YOLO(weights)
    net = model.model.eval()
    target = find_layer(net, layer, first=(layer == "cbam"))
    if target is None:
        raise ValueError(f"模型中未找到 {layer} 层")

    x = torch.from_numpy(
        cv2.resize(img_bgr, (imgsz, imgsz)).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    stored = {}

    def fw(m, inp, out):
        stored["feat"] = out
        out.register_hook(lambda g: stored.__setitem__("grad", g))

    h = target.register_forward_hook(fw)
    try:
        with torch.enable_grad():
            x.requires_grad_(True)
            y = net(x)  # eval 前向： (y,) 或 (y, feats)，y 为 (bs, 4+nc, anchors)
            if isinstance(y, (tuple, list)):
                y = y[0]
            scores = y[0, 4 + int(cls)]
            top = float(scores.max())
            if top < 1e-6:  # 该类无响应时退回全类最大分数（并记录）
                scores = y[0, 4:].max(dim=0).values
                top = float(scores.max())
            scores.max().backward()
    finally:
        h.remove()

    feat = stored["feat"].detach()          # (1, C, h, w)
    grad = stored["grad"].detach()
    w = grad.mean(dim=(2, 3), keepdim=True)  # 通道权重 (1,C,1,1)
    cam = torch.relu((feat * w).sum(dim=1))[0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_np = cv2.resize(cam.numpy(), (img_bgr.shape[1], img_bgr.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)

    info = {"layer": layer, "cls": int(cls), "top_score": round(top, 4),
            "fallback": top < 1e-6, "feat_shape": list(feat.shape)}
    return overlay, cam_np, info


def draw_dets(img_bgr, weights, conf=0.25, imgsz=512):
    """叠加检测框（defect 红 / particle 黄），返回 (图, 检出列表)。"""
    model = YOLO(weights)
    r = model.predict(img_bgr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        c, p = int(b.cls), float(b.conf)
        color = (0, 0, 255) if c == 0 else (0, 165, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, f"{r.names[c]} {p:.2f}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        out.append({"cls": r.names[c], "conf": round(p, 3), "xyxy": [x1, y1, x2, y2]})
    return img_bgr, out
