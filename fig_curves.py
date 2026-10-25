# -*- coding: utf-8 -*-
"""训练曲线图：A0 基线 vs 关键改进实验（box_loss 与 mAP50 随 epoch 变化）。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent / "CoatingDet-Defect-Detection"
OUT = Path(r"D:\毕业论文\04_论文\figures")
OUT.mkdir(parents=True, exist_ok=True)

# 显示名 → runs 子目录
RUNS = {
    "A0 基线": "yolo11n_20ep",
    "C +CBAM": "C_cbam",
    "E +白平衡": "E_gw",
    "F2 = C+E": "F2_C_E",
}
COLORS = {"A0 基线": "#9AA6B2", "C +CBAM": "#0F6E7E", "E +白平衡": "#5C7278", "F2 = C+E": "#E8752A"}


def read(run):
    f = REPO / "runs" / run / "results.csv"
    if not f.exists():
        return None
    epochs, box, m50 = [], [], []
    for row in csv.DictReader(open(f)):
        e = row.get("epoch")
        if e is None:
            continue
        epochs.append(int(e) + 1)
        box.append(float(next(v for k, v in row.items() if k.strip() in ("train/box_loss", "box_loss"))))
        m50.append(float(next(v for k, v in row.items() if k.strip() in ("metrics/mAP50(B)", "metrics/mAP50(B)"))))
    return epochs, box, m50


fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=200)
for name, run in RUNS.items():
    d = read(run)
    if not d:
        print("skip (no csv):", name)
        continue
    ep, box, m50 = d
    axes[0].plot(ep, box, label=name, color=COLORS[name], lw=1.6)
    axes[1].plot(ep, m50, label=name, color=COLORS[name], lw=1.6)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train box_loss")
axes[0].set_title("训练损失收敛对比")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val mAP50")
axes[1].set_title("验证集 mAP50 收敛对比")
for ax in axes:
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig_curves.png")
print("fig_curves.png saved")
