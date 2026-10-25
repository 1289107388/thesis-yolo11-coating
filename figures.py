# -*- coding: utf-8 -*-
"""论文图表生成：数据集统计、框尺寸分布、消融对比（全部来自真实数据）。"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path(r"F:\CoatingDet_v2")
REPO = Path(__file__).resolve().parent / "CoatingDet-Defect-Detection"
OUT = Path(r"D:\毕业论文\04_论文\figures")
OUT.mkdir(parents=True, exist_ok=True)

C_DEFECT, C_PART, C_GRID = "#0F6E7E", "#E8752A", "#D8E2E5"


def count_boxes(labels_dir):
    c = Counter()
    for f in Path(labels_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                c[int(p[0])] += 1
    return c


# ---- 图1：各域切片数与两类框数 ----
domains, tiles_n, defect_n, particle_n = [], [], [], []
for d, tdir in [("train", "train/tiles/train"), ("val", "train/tiles/val"),
                ("test_1", "test_1/tiles"), ("test_2", "test_2/tiles"), ("test_3", "test_3/tiles")]:
    img_dir = DATA / tdir / "images"
    lbl_dir = DATA / tdir / "labels"
    domains.append(d)
    tiles_n.append(len(list(img_dir.glob("*.jpg"))))
    c = count_boxes(lbl_dir)
    defect_n.append(c[0])
    particle_n.append(c[1])

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
x = range(len(domains))
w = 0.28
ax.bar([i - w for i in x], tiles_n, w, label="切片数", color=C_GRID, edgecolor="#9AA6B2")
ax.bar(list(x), defect_n, w, label="defect 框", color=C_DEFECT)
ax.bar([i + w for i in x], particle_n, w, label="particle 框", color=C_PART)
for i, (t, dn, pn) in enumerate(zip(tiles_n, defect_n, particle_n)):
    ax.text(i - w, t + 12, str(t), ha="center", fontsize=8, color="#5C7278")
    ax.text(i, dn + 12, str(dn), ha="center", fontsize=8, color=C_DEFECT)
    ax.text(i + w, pn + 12, str(pn), ha="center", fontsize=8, color=C_PART)
ax.set_xticks(list(x)); ax.set_xticklabels(domains)
ax.set_ylabel("数量")
ax.set_title("官方切片协议下各子集的切片数与两类标注框数")
ax.legend(frameon=False, ncol=3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_dataset_domains.png"); plt.close(fig)
print("fig_dataset_domains.png", dict(zip(domains, zip(tiles_n, defect_n, particle_n))))

# ---- 图2：缺陷四子类框数（官方 README 数字） ----
subs = ["夹杂", "划伤", "污染物", "针孔"]
vals = [621, 84, 44, 19]
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)
bars = ax.bar(subs, vals, color=[C_DEFECT, C_DEFECT, C_DEFECT, C_PART], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=9)
ax.set_ylabel("框数（全数据集）")
ax.set_title("defect 类四个子类框数分布")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_defect_subtypes.png"); plt.close(fig)
print("fig_defect_subtypes.png")

# ---- 图3：切片级目标尺寸分布（sqrt 面积，px） ----
sizes = []
for tdir in ["train/tiles/train", "train/tiles/val", "test_1/tiles", "test_2/tiles", "test_3/tiles"]:
    for f in (DATA / tdir / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                _, _, _, bw, bh = map(float, p)
                sizes.append(((bw * 512) * (bh * 512)) ** 0.5)
import numpy as np
sizes = np.array(sizes)
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
ax.hist(sizes, bins=48, color=C_DEFECT, alpha=0.85)
ax.axvline(32, color=C_PART, ls="--", lw=1.5)
ax.text(33, ax.get_ylim()[1] * 0.9, "小目标阈值 32px", color=C_PART, fontsize=9)
ax.set_xlabel("目标等效边长 $\\sqrt{w\\cdot h}$（像素，512 切片内）")
ax.set_ylabel("实例数")
ax.set_title(f"切片级目标尺寸分布（共 {len(sizes)} 实例，中位数 {np.median(sizes):.0f}px）")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_bbox_sizes.png"); plt.close(fig)
print(f"fig_bbox_sizes.png median={np.median(sizes):.1f} <32px: {(sizes < 32).mean() * 100:.1f}%")

print("ALL_FIGURES_DONE")
