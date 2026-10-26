# -*- coding: utf-8 -*-
"""YOLOv11 检测的 Grad-CAM（网页与论文图共用）。

原理：选定特征层（默认骨干末端 C2PSA；CBAM 模型可选 head 里的 CBAM），
取该类最大置信度锚点分数反传，梯度全局平均作为通道权重，
加权和过 ReLU 得到类激活图，叠回原图。
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


def _find_layer(net, cls_name):
    idxs = [i for i, m in enumerate(net.model) if type(m).__name__ == cls_name]
    return net.model[idxs[-1]] if idxs else None


def gradcam(weights, img_bgr, cls=0, layer="c2psa", imgsz=512):
    """返回 (叠加图 BGR, 激活图 numpy, info dict)。"""
    model = YOLO(weights)
    net = model.model.eval()
    target = _find_layer(net, LAYERS.get(layer, "C2PSA"))
    if target is None:
        raise ValueError(f"模型中未找到 {layer} 层")

    x = torch.from_numpy(
        cv2.resize(img_bgr, (imgsz, imgsz)).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    stored = {}
    h = target.register_forward_hook(lambda m, i, o: stored.__setitem__("feat", o))
    try:
        with torch.enable_grad():
            x.requires_grad_(True)
            y = net(x)  # eval: (bs, 4+nc, anchors)
            scores = y[0, 4 + cls]
            if float(scores.max()) < 1e-6:  # 该类无响应时退回全类最大分
                scores = y[0, 4:].max(dim=0).values
            scores.max().backward()
    finally:
        h.remove()

    feat, grad = stored["feat"], x.grad  # 无用；真实梯度取自 feat 反传
    # 梯度在 backward 时已流入 feat：通过重新注册捕获
    # ——上面 forward hook 只存了 feat；grad 需要在 backward 前挂 tensor hook。
    raise NotImplementedError  # 占位，由 gradcam_v2 替换
