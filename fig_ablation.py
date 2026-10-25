# -*- coding: utf-8 -*-
"""消融对比图：各实验在四个域上的 mAP50 分组柱状图（数据来自官方评测 json）。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent / "CoatingDet-Defect-Detection"
OUT = Path(r"D:\毕业论文\04_论文\figures")

ORDER = ["A0", "B_p2", "C_cbam", "D_wiou", "E_gw", "F_combo", "A0_60ep", "F2_C_E", "F3_C_D_E", "D2_wiou_rclamp"]
LABEL = {"A0": "A0 基线", "B_p2": "B +P2", "C_cbam": "C +CBAM", "D_wiou": "D +WIoU", "E_gw": "E +白平衡",
         "F_combo": "F 全组合", "A0_60ep": "A0@60ep", "F2_C_E": "F2=C+E", "F3_C_D_E": "F3=C+D+E",
         "D2_wiou_rclamp": "D2 WIoU截断"}
DOMAINS = ["val", "test_1", "test_2", "test_3"]


def load():
    merged = {}
    for d in [REPO / "results"] + sorted(REPO.glob("results_*")):
        f = d / "metrics_conf0.001.json"
        if f.exists():
            for exp, subs in json.loads(f.read_text(encoding="utf-8")).items():
                name = "A0" if d.name == "results" else d.name.replace("results_", "")
                merged.setdefault(name, {}).update(subs)
    return merged


m = load()
exps = [k for k in ORDER if k in m and m[k].get("val")]
fig, ax = plt.subplots(figsize=(11, 4.3), dpi=200)
x = np.arange(len(DOMAINS))
w = 0.8 / max(len(exps), 1)
cmap = plt.cm.viridis(np.linspace(0.08, 0.85, len(exps)))
for i, e in enumerate(exps):
    vals = [m[e].get(d, {}).get("overall", {}).get("mAP50", 0) * 100 for d in DOMAINS]
    bars = ax.bar(x + i * w, vals, w, label=LABEL.get(e, e), color=cmap[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=6.5, rotation=90)
ax.set_xticks(x + w * (len(exps) - 1) / 2)
ax.set_xticklabels(["val", "test_1 暖色", "test_2", "test_3"])
ax.set_ylabel("mAP50 (%)")
ax.set_ylim(0, 100)
ax.set_title("消融实验与组合实验在四个评测域上的 mAP50")
ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight")
print("fig_ablation.png:", exps)
