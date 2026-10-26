# -*- coding: utf-8 -*-
"""论文扩版配图：结构示意 / CBAM 模块 / WIoU r(β) 曲线 / 混淆对比 / 数据样例 / 难例对比。"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "CoatingDet-Defect-Detection"
OUT = Path(r"D:\毕业论文\04_论文\figures")
TEAL, ORANGE, GREY, DARK, LINE = "#0F6E7E", "#E8752A", "#8A9BA3", "#17323A", "#D8E2E5"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=TEAL, fs=9, lw=1.4):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=DARK)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.2))


# ---- 图: YOLOv11 + CBAM 改进结构示意 ----
def fig_arch():
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=200)
    ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 7)
    box(ax, 0.3, 4.6, 2.1, 1.6, "输入\n512×512", "#EAF3F5")
    box(ax, 3.0, 4.6, 3.4, 1.6, "Backbone\nConv×2 + C3k2×4\n+ SPPF + C2PSA")
    box(ax, 7.0, 4.6, 3.2, 1.6, "Neck (PAN)\n自顶向下 + 自底向上\nP2/P3/P4/P5 多尺度")
    box(ax, 10.8, 5.2, 1.7, 1.0, "CBAM\n×3", "#FFE8D9", ORANGE, lw=1.8)
    box(ax, 10.8, 3.6, 1.7, 1.0, "CBAM\n×3", "#FFE8D9", ORANGE, lw=1.8)
    box(ax, 10.8, 2.0, 1.7, 1.0, "CBAM\n×3", "#FFE8D9", ORANGE, lw=1.8)
    box(ax, 13.0, 2.0, 1.6, 4.2, "解耦\n检测头\nP3/P4/P5", "#EAF3F5")
    arrow(ax, 2.4, 5.4, 3.0, 5.4); arrow(ax, 6.4, 5.4, 7.0, 5.4)
    arrow(ax, 10.2, 5.4, 10.8, 5.7); arrow(ax, 10.2, 5.0, 10.8, 4.1); arrow(ax, 10.2, 4.7, 10.8, 2.5)
    arrow(ax, 12.5, 5.7, 13.0, 5.0); arrow(ax, 12.5, 4.1, 13.0, 4.1); arrow(ax, 12.5, 2.5, 13.0, 3.2)
    ax.text(11.65, 6.5, "本文插入位置", ha="center", fontsize=9, color=ORANGE)
    ax.text(4.7, 3.6, "损失端：CIoU → WIoUv3（β 离群度动态聚焦 + r≥0.5 截断）", fontsize=9.5,
            color=DARK, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF6EF", ec=ORANGE, lw=1.2))
    ax.text(4.7, 2.4, "输入端：灰世界白平衡（离线）+ hsv_h 0.015→0.1（在线增强）", fontsize=9.5,
            color=DARK, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#EAF3F5", ec=TEAL, lw=1.2))
    fig.tight_layout(); fig.savefig(OUT / "fig_arch.png", bbox_inches="tight"); plt.close(fig)


# ---- 图: CBAM 模块结构 ----
def fig_cbam():
    fig, ax = plt.subplots(figsize=(9.6, 3.2), dpi=200)
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    box(ax, 0.2, 1.9, 1.5, 1.2, "输入\nF∈R^(C×H×W)")
    box(ax, 2.4, 3.0, 4.6, 1.5, "通道注意力 Mc\nAvgPool/MaxPool → 共享MLP → σ", "#EAF3F5")
    box(ax, 7.6, 3.0, 1.4, 1.5, "⊗", "#FFFFFF", TEAL, fs=13)
    box(ax, 2.4, 0.6, 4.6, 1.5, "空间注意力 Ms\n[mean,max] 通道拼接 → 7×7Conv → σ", "#EAF3F5")
    box(ax, 7.6, 0.6, 1.4, 1.5, "⊗", "#FFFFFF", TEAL, fs=13)
    box(ax, 9.6, 1.9, 1.6, 1.2, "输出\nF′")
    arrow(ax, 1.7, 2.8, 2.4, 3.6); arrow(ax, 7.0, 3.75, 7.6, 3.75)
    arrow(ax, 9.0, 3.75, 9.3, 2.8)  # to spatial
    ax.plot([9.3, 9.3, 2.2], [2.8, 1.35, 1.35], color=GREY, lw=1.2)
    arrow(ax, 2.2, 1.35, 2.4, 1.35)
    arrow(ax, 7.0, 1.35, 7.6, 1.35); arrow(ax, 9.0, 1.35, 9.6, 2.2)
    fig.tight_layout(); fig.savefig(OUT / "fig_cbam.png", bbox_inches="tight"); plt.close(fig)


# ---- 图: WIoU 梯度分配系数 r(β) ----
def fig_wiou():
    import numpy as np
    beta = np.linspace(0.05, 8, 400)
    alpha, delta = 2.0, 3.0
    r = beta / (delta * alpha ** (beta - delta))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    ax.plot(beta, r, color=TEAL, lw=2, label="r(β) = β / (δ·α^(β−δ)), α=2, δ=3")
    ax.axhline(0.5, color=ORANGE, ls="--", lw=1.4)
    ax.axhline(1.0, color=GREY, ls=":", lw=1.2)
    beta2 = np.linspace(0.05, 8, 400)
    r2 = np.clip(beta2 / (delta * alpha ** (beta2 - delta)), 0.5, None)
    ax.plot(beta2, r2, color=ORANGE, lw=1.6, ls="-.", label="D2：r≥0.5 截断")
    ax.set_xlabel("离群度 β = (1−Ī)/(1−IoU)")
    ax.set_ylabel("梯度分配系数 r")
    ax.set_title("WIoUv3 梯度分配系数：放大一般离群样本、衰减极端样本")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_wiou_r.png"); plt.close(fig)


# ---- 图: 混淆计数对比（真实评测 json） ----
def fig_confusion():
    def conf(exp_dir, exp, dom):
        p = REPO / exp_dir / "metrics_conf0.001.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        return d[exp][dom]["confusion"]
    a0 = conf("results", "YOLOv11n", "test1")
    c = conf("results_C_cbam", "C_cbam", "test1")
    e = conf("results_E_gw", "E_gw", "test1")
    r = conf("results", "YOLOv11n", "test3")
    rc = conf("results_C_cbam", "C_cbam", "test3")
    groups = [
        ("test_1 粒子误报为缺陷", [a0["particle_as_defect"], c["particle_as_defect"], e["particle_as_defect"]]),
        ("test_1 粒子漏检", [a0["particle_missed"], c["particle_missed"], e["particle_missed"]]),
        ("test_3 粒子误报为缺陷", [r["particle_as_defect"], rc["particle_as_defect"], None]),
        ("test_3 缺陷漏检", [r["defect_missed"], rc["defect_missed"], None]),
    ]
    labels = ["A0 基线", "C +CBAM", "E +白平衡"]
    colors = [GREY, TEAL, ORANGE]
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=200)
    x = np.arange(len(groups)); w = 0.24
    for i, lab in enumerate(labels):
        vals = [g[1][i] if (len(g[1]) > i and g[1][i] is not None) else 0 for g in groups]
        pos = x + (i - 1) * w
        bars = ax.bar(pos, vals, w, label=lab, color=colors[i])
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("实例数")
    ax.set_title("两类混淆的错检计数（conf=0.001 工作点，E 仅在含其训练配置的域上报告）")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion.png"); plt.close(fig)


# ---- 图: 数据集切片样例（真实图片 + GT 框） ----
def fig_samples():
    import cv2
    DATA = Path(r"F:\CoatingDet_v2")
    picks = []
    for dom in ["train", "test_1", "test_2", "test_3"]:
        tdir = DATA / dom / ("tiles/train" if dom == "train" else "tiles")
        imgs, lbls = tdir / "images", tdir / "labels"
        scored = sorted(((sum(1 for l in f.read_text().splitlines() if l.strip()), f.stem)
                         for f in lbls.glob("*.txt")), reverse=True)
        if scored:
            picks.append((dom, imgs / f"{scored[min(1, len(scored) - 1)][1]}.jpg",
                          lbls / f"{scored[min(1, len(scored) - 1)][1]}.txt"))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=170)
    for ax, (dom, img_p, lbl_p) in zip(axes, picks):
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        for line in lbl_p.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cls, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
            x1, y1 = (xc - bw / 2) * 512, (yc - bh / 2) * 512
            color = "#FF3B30" if cls == 0 else "#3B8BFF"
            ax.add_patch(Rectangle((x1, y1), bw * 512, bh * 512, ec=color, fc="none", lw=1.6))
        ax.set_title(f"{dom}（红=defect 蓝=particle）", fontsize=9)
        ax.axis("off")
    fig.suptitle("官方切片协议下的真实样例与标注", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_samples.png", bbox_inches="tight"); plt.close(fig)


# ---- 图: 难例对比 A0 vs E（test_1 同图，真实推理） ----
def fig_hardcases():
    import cv2
    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO
    DATA = Path(r"F:\CoatingDet_v2\test_1\tiles")
    lbl = DATA / "labels"
    best = sorted(((sum(1 for l in f.read_text().splitlines() if l.strip()), f.stem)
                   for f in lbl.glob("*.txt")), reverse=True)[2][1]
    img_path = DATA / "images" / f"{best}.jpg"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), dpi=170)
    for ax, (wname, wpath) in zip(axes, [
            ("A0 基线：大量灰尘被误判为 defect", REPO / "runs/yolo11n_20ep/weights/best.pt"),
            ("E +白平衡：误报被抑制", REPO / "runs/E_gw/weights/best.pt")]):
        m = YOLO(str(wpath))
        r = m.predict(str(img_path), imgsz=512, conf=0.25, device="cpu", verbose=False)[0]
        ax.imshow(r.plot()[:, :, ::-1]); ax.set_title(wname, fontsize=10); ax.axis("off")
    fig.suptitle(f"暖色域 test_1 同一切片的检测对比（{best[:26]}…）", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_hardcases.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_arch(); print("fig_arch.png")
    fig_cbam(); print("fig_cbam.png")
    fig_wiou(); print("fig_wiou_r.png")
    fig_confusion(); print("fig_confusion.png")
    fig_samples(); print("fig_samples.png")
    fig_hardcases(); print("fig_hardcases.png")
    print("FIGS2_DONE")
