# WIoUv3 边界框回归损失（Tong et al., 2023, arXiv:2301.10051）
# 用法：import wiou; wiou.apply()  （在 YOLO().train 之前调用）
# L_WIoUv3 = r · R_WIoU · L_IoU
#   R_WIoU = exp(d2 / c2)      距离注意力，整体 detach（论文 * 号：不产生梯度）
#   β = L̄*/L*                  离群度（当前 IoU 均值 EMA 相对每个框）
#   r = β / (δ·α^(β−δ))        梯度分配系数（α=2, δ=3），放大多数离群样本、
#                               削减极端低质量样本的有害梯度
import torch
import torch.nn as nn

import ultralytics.utils.loss as _loss_mod

ALPHA = 2.0
DELTA = 3.0


def _wiou_iou(pb, tb):
    """xyxy 框的 (iou, d2, c2)。pb/tb: (N,4)"""
    # 交并比
    inter_x1, inter_y1 = torch.maximum(pb[:, 0], tb[:, 0]), torch.maximum(pb[:, 1], tb[:, 1])
    inter_x2, inter_y2 = torch.minimum(pb[:, 2], tb[:, 2]), torch.minimum(pb[:, 3], tb[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_p = (pb[:, 2] - pb[:, 0]).clamp(0) * (pb[:, 3] - pb[:, 1]).clamp(0)
    area_t = (tb[:, 2] - tb[:, 0]).clamp(0) * (tb[:, 3] - tb[:, 1]).clamp(0)
    union = area_p + area_t - inter
    iou = inter / union.clamp(1e-9)
    # 中心距² 与 最小外接框对角线²
    pcx, pcy = (pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2
    tcx, tcy = (tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2
    d2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    cw = torch.maximum(pb[:, 2], tb[:, 2]) - torch.minimum(pb[:, 0], tb[:, 0])
    chh = torch.maximum(pb[:, 3], tb[:, 3]) - torch.minimum(pb[:, 1], tb[:, 1])
    c2 = cw**2 + chh**2
    return iou, d2, c2


class WIoUBboxLoss(_loss_mod.BboxLoss):
    def __init__(self, reg_max: int = 16, momentum: float = 0.9):
        super().__init__(reg_max)
        self.iou_mean = 1.0          # IoU 均值 EMA（离群度基准）
        self.momentum = momentum

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        iou, d2, c2 = _wiou_iou(pb, tb)
        with torch.no_grad():
            self.iou_mean = (self.momentum * self.iou_mean
                             + (1 - self.momentum) * iou.mean().item())
            r_wiou = torch.exp(d2 / c2.clamp(1e-9))          # 距离注意力，无梯度
            beta = (1.0 - self.iou_mean) / (1.0 - iou + 1e-9)  # 离群度 β
            r = beta / (DELTA * ALPHA ** (beta - DELTA))     # 梯度分配系数，无梯度
        loss_iou = ((r * r_wiou * (1.0 - iou)) * weight).sum() / target_scores_sum

        # DFL 部分与原实现一致
        if self.dfl_loss:
            from ultralytics.utils.tal import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


def apply():
    """把 v8DetectionLoss 内的 BboxLoss 换成 WIoU 版（模块级名字替换）。"""
    _loss_mod.BboxLoss = WIoUBboxLoss
