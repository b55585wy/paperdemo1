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
    # 确保损失函数正确处理维度不匹配
    # 在损失函数中添加维度调整
    def forward(self, pred, target):
        # 记录原始形状
        logging.info(f"Loss function - pred shape: {pred.shape}, target shape: {target.shape}")
        
        # 调整维度
        batch_size, seq_len = pred.shape[0], pred.shape[1]
        num_classes = pred.shape[2]
        
        # 将预测从 [batch, seq, classes, 1, 1] 重塑为 [batch*seq, classes]
        # 使用 reshape 而不是 view，因为 view 要求内存连续
        pred = pred.reshape(batch_size * seq_len, num_classes)
        
        # 将目标从 [batch, seq, 1, 1, 1] 重塑为 [batch*seq]
        target = target.reshape(-1)  # 使用 reshape 而不是 view
        
        # 记录调整后的形状
        logging.info(f"Reshaped - pred: {pred.shape}, target: {target.shape}")
        
        # 计算损失
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