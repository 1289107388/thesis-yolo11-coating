# -*- coding: utf-8 -*-
"""实验 E 预处理：对官方切片训练集做灰世界白平衡，生成 tiles_gw（标签原样复制）。"""
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "CoatingDet-Defect-Detection" / "src"))
from coatingdet.whitebalance import grey_world  # noqa: E402

ROOT = Path(r"F:\CoatingDet_v2\train\tiles")
DST = Path(r"F:\CoatingDet_v2\train\tiles_gw")


def main():
    for split in ["train", "val"]:
        img_src, lbl_src = ROOT / split / "images", ROOT / split / "labels"
        img_dst, lbl_dst = DST / split / "images", DST / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(img_src.iterdir()):
            img = cv2.imread(str(p))
            if img is None:
                continue
            cv2.imwrite(str(img_dst / p.name), grey_world(img))
            lbl = lbl_src / f"{p.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, lbl_dst / lbl.name)
            n += 1
        print(f"{split}: {n} images -> {img_dst}")


if __name__ == "__main__":
    main()
