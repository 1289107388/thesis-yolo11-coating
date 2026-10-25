# -*- coding: utf-8 -*-
"""汇总全部实验 → thesis_data.json（论文与网页共用，含参数量统计）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "CoatingDet-Defect-Detection"
OUT = ROOT / "thesis_data.json"

RUNS = {  # 显示名 -> (runs 子目录, 参数说明)
    "A0": ("yolo11n_20ep", "YOLOv11n 官方协议基线"),
    "B_p2": ("B_p2", "B：+P2 小目标检测头"),
    "C_cbam": ("C_cbam", "C：+CBAM 注意力"),
    "D_wiou": ("D_wiou", "D：+WIoUv3 损失"),
    "E_gw": ("E_gw", "E：+灰世界白平衡与宽色相增强"),
    "F_combo": ("F_combo", "F：B+C+D+E 全组合"),
    "A0_60ep": ("A0_60ep", "A0 基线 @ 60 epoch"),
    "F2_C_E": ("F2_C_E", "F2：C+E"),
    "F3_C_D_E": ("F3_C_D_E", "F3：C+D+E"),
    "D2_wiou_rclamp": ("D2_wiou_rclamp", "D2：WIoU + r≥0.5 截断"),
    "v5n_comp": ("v5n_comp", "YOLOv5nu 对比"),
    "v8n_comp": ("v8n_comp", "YOLOv8n 对比"),
    "v10n_comp": ("v10n_comp", "YOLOv10n 对比"),
    "rtdetr": ("rtdetr_l_20ep", "RT-DETR-L 对比"),
    "C_60ep": ("C_60ep", "本文最终候选：C @ 60 epoch"),
    "F2_60ep": ("F2_60ep", "本文部署变体：C+E @ 60 epoch"),
}


def main():
    merged = {}
    for d in [REPO / "results"] + sorted(REPO.glob("results_*")):
        f = d / "metrics_conf0.001.json"
        if f.exists():
            for exp, subs in json.loads(f.read_text(encoding="utf-8")).items():
                name = "A0" if d.name == "results" else d.name.replace("results_", "")
                merged.setdefault(name, {}).update(subs)

    out = {"meta": {"date": "2026-10-26", "protocol": "epochs20/batch8/imgsz512/seed0, eval conf0.001 iou0.7"}, "exp": {}}
    for key, (run, desc) in RUNS.items():
        if key not in merged:
            continue
        entry = {"desc": desc}
        for dom in ["val", "test_1", "test_2", "test_3"]:
            src = dom if dom in merged[key] else dom.replace("_", "")  # 官方 json 用 test1 无下划线
            m = merged[key].get(src, {}).get("overall")
            if m:
                entry[dom] = {"P": round(m["P"], 3), "R": round(m["R"], 3),
                              "mAP50": round(m["mAP50"], 3), "mAP50_95": round(m["mAP50_95"], 3)}
        out["exp"][key] = entry

    # 参数量与 GFLOPs（读权重）
    sys.path.insert(0, str(ROOT))
    import cbam  # noqa
    import ultralytics.nn.tasks as tasks
    tasks.CBAM = cbam.CBAM
    from ultralytics import YOLO
    for key, (run, _) in RUNS.items():
        w = REPO / "runs" / run / "weights" / "best.pt"
        if w.exists():
            try:
                m = YOLO(str(w))
                out["exp"][key]["params_M"] = round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2)
            except Exception as e:
                out["exp"][key]["params_err"] = str(e)[:60]

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    for k, v in out["exp"].items():
        print(k, v.get("desc"), "val", v.get("val", {}).get("mAP50"), "t3", v.get("test_3", {}).get("mAP50"),
              v.get("params_M", ""), "M")


if __name__ == "__main__":
    main()
