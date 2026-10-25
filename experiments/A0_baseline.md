# A0 · YOLOv11n 官方协议基线（2026-10-25）

数据：CoatingDet v2 官方切片协议（512 / 无重叠 / min_bbox 0.35 / min_tile 0.01 / 空片丢弃），train 1,156 张含目标切片，val 30 张（Low-Cold 会话，官方 val_prefix 路由）。

训练：`python -m coatingdet.train --model yolo11n --epochs 20 --name yolo11n_20ep`
协议：batch=8, imgsz=512, seed=0, patience=1000（关早停）, AMP, AdamW lr0≈0.00167。
环境：conda base（Python 3.13.5, torch 2.6.0+cu124, ultralytics **8.3.240**；官方 pin 8.3.100，同 8.3 系列，数字可能有小幅版本差）。
硬件：RTX 3060 Laptop 6GB，20 epoch 约 8 分钟，无 NaN。

评测：conf=0.001, NMS iou=0.7（AP 定义工作点）。

| 域 | P | R | mAP50 | mAP50-95 | 说明 |
|---|---|---|---|---|---|
| val | 0.888 | 0.788 | 0.820 | 0.444 | 33 实例全 defect |
| test_1 (Low-Warm) | 0.462 | 0.917 | 0.711 | 0.494 | 仅 particle；particle→defect 误报 175 |
| test_2 (Low-Cold) | 0.733 | 0.742 | 0.719 | 0.302 | defect→particle 5 |
| test_3 (High-Cold) | 0.757 | 0.747 | 0.816 | 0.508 | defect 0.736/0.375, particle 0.895/0.640；particle→defect 144 |

模型：2,582,542 参数，6.3 GFLOPs，推理 2.3–3.7 ms/张（GPU）。

结论（论文素材）：
1. 暖色域 test_1 精度崩到 0.462——颜色是 defect/particle 的主判别线索，跨色温泛化是真实痛点（官方论文结论一致）。
2. 两类混淆方向主要是 particle→defect（把灰尘当缺陷），工程代价是误报返修。
3. defect 类 mAP50-95（0.375~0.444）显著低于 particle——框回归精度差，小目标（针孔）定位难。

产物：`../03_实验/A0_baseline/`（results.csv / args.yaml / metrics json），权重 `../03_实验/weights/yolo11n_20ep_best.pt`，官方评测表在 `CoatingDet-Defect-Detection/results/`。

下一步消融：B +P2 检测头 · C +注意力 · D +WIoU · E +白平衡/增强 · F 组合。
