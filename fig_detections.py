# -*- coding: utf-8 -*-
"""检测可视化图：从三个测试域各取若干切片，用指定模型推理并绘制检测框网格图。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbam  # noqa: F401
import ultralytics.nn.tasks as tasks
tasks.CBAM = cbam.CBAM
from ultralytics import YOLO  # noqa: E402

DATA = Path(r"F:\CoatingDet_v2")
OUT = Path(r"D:\毕业论文\04_论文\figures")


def best_tiles(domain, k=2, min_boxes=2):
    scored = []
    for lbl in (DATA / domain / "tiles" / "labels").glob("*.txt"):
        n = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        if n >= min_boxes:
            scored.append((n, lbl.stem))
    scored.sort(reverse=True)
    return [s for _, s in scored[:k]]


def main(weights, tag, conf=0.2):
    model = YOLO(weights)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), dpi=170)
    picks = [(d, s) for d in ["test_1", "test_2", "test_3"] for s in best_tiles(d, 2)]
    for ax, (d, stem) in zip(axes.flat, picks):
        img_path = DATA / d / "tiles" / "images" / f"{stem}.jpg"
        r = model.predict(str(img_path), imgsz=512, conf=conf, device="cpu", verbose=False)[0]
        ax.imshow(r.plot()[:, :, ::-1])
        ax.set_title(f"{d} · {stem[:22]}…", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"检测可视化（{tag}，conf={conf}）：红=defect 需返修，其他=particle", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_detections_{tag}.png", bbox_inches="tight")
    print(f"fig_detections_{tag}.png saved")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "model",
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.2)
