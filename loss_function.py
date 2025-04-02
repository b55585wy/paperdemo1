import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        # Move weights to the same device as the model
        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight))
        else:
            self.weight = None
    def forward(self, pred, target):
        # 记录形状以便调试
        logging.info(f"Loss function - pred shape: {pred.shape}, target shape: {target.shape}")
        
        # 重塑预测值从 [batch, seq, classes, 1, 1] 到 [batch*seq, classes]
        batch_size, seq_len, n_classes = pred.size(0), pred.size(1), pred.size(2)
        pred = pred.view(batch_size * seq_len, n_classes, -1).squeeze(-1)
        
        # 重塑目标从 [batch, seq, 1, 1, 1] 到 [batch*seq]
        target = target.view(batch_size * seq_len)
        
        # 确保目标是长整型
        target = target.long()
        
        # 权重会自动与预测值在同一设备上
        return F.cross_entropy(pred, target, weight=self.weight)