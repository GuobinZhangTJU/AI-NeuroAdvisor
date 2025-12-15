import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        # 标记
        if input_tensor.shape[0] != target_tensor.shape[0]:
            # 修改第二个张量的第一个维度，使其与第一个张量的第一个维度相同
            target_tensor = target_tensor[:input_tensor.shape[0]]
        log_prob = F.log_softmax(input_tensor, dim=-1)
        target_tensor = target_tensor.long()
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction)