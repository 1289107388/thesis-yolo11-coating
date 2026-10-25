# -*- coding: utf-8 -*-
"""汇总所有实验的官方评测 json → webapp/static/results.json。"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent / "CoatingDet-Defect-Detection"
OUT = Path(__file__).resolve().parent / "webapp" / "static" / "results.json"

merged = {}
for d in [REPO / "results"] + sorted(REPO.glob("results_*")):
    f = d / "metrics_conf0.001.json"
    if not f.exists():
        continue
    data = json.loads(f.read_text(encoding="utf-8"))
    for exp, subsets in data.items():
        name = d.name.replace("results_", "") if d.name != "results" else "A0"
        if name != "A0":
            merged.setdefault(name, {}).update(subsets)
        else:
            merged.setdefault("A0", {}).update(subsets)

OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, "experiments:", sorted(merged.keys()))
