# -*- coding: utf-8 -*-
"""汇总全部实验 → webapp/static/results.json（含逐域 P/R、逐类、混淆计数、参数量）。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "CoatingDet-Defect-Detection"
TD = ROOT / "thesis_gen" / "thesis_data.json"
OUT = ROOT / "webapp" / "static" / "results.json"

LABEL = {
    "A0": "A0 基线 yolo11n", "B_p2": "B +P2 头", "C_cbam": "C +CBAM", "D_wiou": "D +WIoUv3",
    "E_gw": "E +白平衡/宽色相", "F_combo": "F 全组合", "A0_60ep": "A0@60ep", "F2_C_E": "F2=C+E",
    "F3_C_D_E": "F3=C+D+E", "D2_wiou_rclamp": "D2 WIoU+截断", "v5n_comp": "YOLOv5nu",
    "v8n_comp": "YOLOv8n", "v10n_comp": "YOLOv10n", "rtdetr": "RT-DETR-L", "C_60ep": "C@60ep",
    "F2_60ep": "F2@60ep",
}
ORDER = ["A0", "B_p2", "C_cbam", "D_wiou", "D2_wiou_rclamp", "E_gw", "F_combo",
         "A0_60ep", "F2_C_E", "F3_C_D_E", "C_60ep", "F2_60ep",
         "v5n_comp", "v8n_comp", "v10n_comp", "rtdetr"]


def main():
    merged = {}
    for d in [REPO / "results"] + sorted(REPO.glob("results_*")):
        f = d / "metrics_conf0.001.json"
        if not f.exists():
            continue
        for exp, subs in json.loads(f.read_text(encoding="utf-8")).items():
            name = "A0" if d.name == "results" else d.name.replace("results_", "")
            merged.setdefault(name, {}).update(subs)

    params = {}
    if TD.exists():
        for k, v in json.loads(TD.read_text(encoding="utf-8"))["exp"].items():
            params[k] = v.get("params_M")

    out = []
    for key in ORDER:
        if key not in merged:
            continue
        e = {"id": key, "label": LABEL.get(key, key), "params_M": params.get(key), "domains": {}}
        for raw, dom in [("val", "val"), ("test1", "test_1"), ("test2", "test_2"), ("test3", "test_3")]:
            s = merged[key].get(raw)
            if not s:
                continue
            pc = s.get("per_class", {})
            e["domains"][dom] = {
                "P": round(s["overall"]["P"] * 100, 1), "R": round(s["overall"]["R"] * 100, 1),
                "mAP50": round(s["overall"]["mAP50"] * 100, 1),
                "mAP50_95": round(s["overall"]["mAP50_95"] * 100, 1),
                "per_class": {c: {k2: round(v2 * 100, 1) for k2, v2 in m.items() if k2 in ("P", "R", "mAP50", "mAP50_95")}
                              for c, m in pc.items()} if pc else {},
                "confusion": s.get("confusion", {}),
            }
        ds = e["domains"]
        if all(d in ds for d in ("val", "test_1", "test_2", "test_3")):
            e["mean4"] = round(sum(ds[d]["mAP50"] for d in ("val", "test_1", "test_2", "test_3")) / 4, 1)
        out.append(e)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, len(out), "experiments")


if __name__ == "__main__":
    main()
