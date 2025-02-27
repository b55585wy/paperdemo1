import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else None
        
    def forward(self, pred, target):
        return F.cross_entropy(pred, target, weight=self.weight)