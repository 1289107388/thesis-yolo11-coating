# -*- coding: utf-8 -*-
"""统一实验训练入口：官方协议 + 可选改进项。

用法示例（都在 02_代码 目录下执行，PYTHONPATH 含官方 src 与本目录）：
  B: python exp_train.py --model models/yolo11n-p2.yaml --name B_p2
  C: python exp_train.py --model models/yolo11n-cbam.yaml --name C_cbam
  D: python exp_train.py --model yolo11n.pt          --name D_wiou --loss wiou
  E: python exp_train.py --model yolo11n.pt          --name E_gw --data configs_own/data_val_gw.yaml --hsv-h 0.1
  F: python exp_train.py --model models/yolo11n-p2-cbam.yaml --name F_combo --loss wiou \
        --data configs_own/data_val_gw.yaml --hsv-h 0.1

固定协议（与官方一致）：epochs 20 / batch 8 / imgsz 512 / seed 0 / patience 1000 / AMP。
"""
import argparse
from pathlib import Path

import ultralytics.nn.tasks as _tasks
from ultralytics import YOLO

import cbam  # noqa: F401  导入即注册
_tasks.CBAM = cbam.CBAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yaml 或 .pt")
    ap.add_argument("--weights", default=None, help="预训练权重（yaml 建模时迁移）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=None, help="data yaml，默认官方 configs/data_val.yaml")
    ap.add_argument("--loss", choices=["ciou", "wiou"], default="ciou")
    ap.add_argument("--wiou-alpha", type=float, default=None)
    ap.add_argument("--wiou-delta", type=float, default=None)
    ap.add_argument("--wiou-rmin", type=float, default=None)
    ap.add_argument("--wiou-rmax", type=float, default=None)
    ap.add_argument("--hsv-h", type=float, default=None, help="色相增强范围（官方默认 0.015）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    repo = Path(__file__).resolve().parent / "CoatingDet-Defect-Detection"
    data = a.data or str(repo / "configs" / "data_val.yaml")

    if a.loss == "wiou":
        import wiou
        wiou.apply(alpha=a.wiou_alpha, delta=a.wiou_delta, r_min=a.wiou_rmin, r_max=a.wiou_rmax)
        print(f"[exp_train] WIoUv3: alpha={a.wiou_alpha or 2.0} delta={a.wiou_delta or 3.0} "
              f"r_min={a.wiou_rmin} r_max={a.wiou_rmax}")

    model = YOLO(a.model)
    if a.weights:
        model.load(a.weights)
        print(f"[exp_train] 已迁移预训练权重: {a.weights}")

    extra = {}
    if a.hsv_h is not None:
        extra["hsv_h"] = a.hsv_h
        print(f"[exp_train] hsv_h = {a.hsv_h}（默认 0.015）")

    model.train(
        data=data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, seed=a.seed,
        patience=1000, plots=True, val=True, amp=True,
        project=str(repo / "runs"), name=a.name, exist_ok=True, **extra,
    )
    print(f"[exp_train] weights: {repo / 'runs' / a.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
