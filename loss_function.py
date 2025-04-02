import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import f1_score
import numpy as np

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

def calculate_f1_score(pred, target, average='macro'):
    """
    计算F1分数
    
    参数:
    - pred: 模型预测值，形状为 [batch, seq, classes, ...] 或 [batch, classes, ...]
    - target: 目标标签，形状为 [batch, seq, ...] 或 [batch, ...]
    - average: 平均方法，可选 'micro', 'macro', 'weighted', 'samples' 或 None
    
    返回:
    - f1: F1分数
    """
    # 处理多维预测值
    if len(pred.shape) > 3:  # 5D tensor [batch, seq, classes, 1, 1]
        batch_size, seq_len, n_classes = pred.size(0), pred.size(1), pred.size(2)
        pred = pred.view(batch_size * seq_len, n_classes, -1).squeeze(-1)
        target = target.view(-1)
    elif len(pred.shape) > 2:  # 3D tensor [batch, seq, classes]
        batch_size, seq_len = pred.size(0), pred.size(1)
        pred = pred.view(batch_size * seq_len, -1)
        target = target.view(batch_size * seq_len)
    
    # 获取预测的类别
    _, predicted = torch.max(pred, 1)
    
    # 转换为numpy数组以使用sklearn
    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()
    
    # 计算F1分数
    return f1_score(target, predicted, average=average)