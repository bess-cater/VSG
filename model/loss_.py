import torch
import torch.nn.functional as F
from torch import nn

#https://github.com/wjun0830/QD-DETR/blob/main/qd_detr/span_utils.py#L91
class Loss_(nn.Module):
    def __init__(self):
        super().__init__()
        self.giou = self.generalized_temporal_iou()
        self.l1 = F.l1_loss()
    def generalized_temporal_iou(self, spans1, spans2):
        spans1 = spans1.float()
        spans2 = spans2.float()
        assert (spans1[:, 1] >= spans1[:, 0]).all()
        assert (spans2[:, 1] >= spans2[:, 0]).all()
        iou, union = temporal_iou(spans1, spans2)

        left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
        right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
        enclosing_area = (right - left).clamp(min=0)  # (N, M)

        return iou - (enclosing_area - union) / enclosing_area
    
    def forward(self, outputs, targets):
        l1 = self.l1(targets, outputs, reduction='none')
        giou = self.generalized_temporal_iou(targets, outputs)
        loss=l1+giou
        return loss