import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ConbimambaBlock

class SleepConmambaModel(nn.Module):
    """
    基于Conmamba的睡眠阶段分类模型
    
    使用ConbimambaBlock作为主要构建块，添加输入投影层和输出分类层
    """
    def __init__(
            self,
            input_dim: int = 1,  # 输入通道数（EEG信号）
            hidden_dim: int = 64,  # 隐藏层维度
            num_classes: int = 5,  # 睡眠阶段分类数量
            num_blocks: int = 2,  # Conmamba块的数量
            **block_params  # ConbimambaBlock的其他参数
        ):
        super(SleepConmambaModel, self).__init__()
        
        # 输入投影层：将输入信号投影到隐藏维度
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Conmamba块堆叠
        self.blocks = nn.ModuleList([
            ConbimambaBlock(
                encoder_dim=hidden_dim,
                **block_params
            ) for _ in range(num_blocks)
        ])
        
        # 输出分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 [B, T, C]，其中：
               B: 批次大小
               T: 序列长度
               C: 通道数
        
        Returns:
            输出预测，形状为 [B, T, num_classes]
        """
        # 确保输入形状正确
        batch_size, seq_len, channels = x.shape
        
        # 转换为卷积层所需的形状 [B, C, T]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 转换回 [B, T, C] 用于Conmamba块
        x = x.transpose(1, 2)
        
        # 通过Conmamba块
        for block in self.blocks:
            x = block(x)
        
        # 分类
        logits = self.classifier(x)
        
        return logits