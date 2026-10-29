# 风机塔筒涂层缺陷检测

- 题目：基于 YOLOv11 的风机塔筒涂层缺陷检测
- 数据集：CoatingDet v2（Zenodo DOI `10.5281/zenodo.21600391`，15.2 GB


## 环境

| 用途 | 环境 | 说明 |
|---|---|---|
| 日常实验 | conda base（`F:\ana`，Python 3.13.5） | torch 2.6.0+cu124（CUDA 可用）、ultralytics 8.3.240、opencv 4.12、numpy 2.5、openpyxl |
| 官方精确复现 | conda `yolo`（`F:\ana\envs\yolo`，Python 3.11  | 按官方 requirements.txt：`ultralytics==8.3.100`、`numpy<2`、`opencv>=4.7,<4.12`；8.4 系列别升（RT-DETR 后处理变了） |

- 驱动 610.62（支持 CUDA 13.3）；pip 版 torch 自带 CUDA 运行时，无需系统 CUDA Toolkit。

## 目录

```
CoatingDet-Defect-Detection/   官方代码（克隆，只读参考，勿改；切片/训练/评测用它的模块）
download_coatingdet.py         Zenodo 断点续传下载器（备用：手动下载失败时可用）
```

```
datasets/
├── metadata.csv        # 分号分隔，每图一行，5,416 行
├── train/raw_data/{images,labels}/     1,154 张（1,128 训练 + 26 验证）
├── test_1/raw_data/...                 138 张 Low-Warm（只有 particle）
├── test_2/raw_data/...                 349 张 Low-Cold
└── test_3/raw_data/...                 3,775 张 High-Cold
```

## 官方协议速记（复现基准用）

- 切片（`configs/tiling.yaml`）：tile 512 / overlap 0 / min_bbox_fraction 0.35 / min_tile_fraction 0.01 / keep_empty false；`val_prefix "Image__2024-05-27"` 把 Low-Cold 会话路由为 val
- 训练（`coatingdet.train`）：YOLOv11n，epochs=20、batch=8、imgsz=512、seed=0、无早停，AdamW lr0=0.00167
- 评测：imgsz=512、conf=0.001、NMS iou=0.7；部署工作点 conf=0.25 另报 P/R
- 一键复现：`bash scripts/reproduce_all.sh`（约 1 GPU·小时，跑前先 `export PYTHONPATH="$PWD/src"`
