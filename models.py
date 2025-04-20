import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """位置编码模块，为序列添加位置信息"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]

class ChannelAttentionModule(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MambaSS1D(nn.Module):
    """
    一维状态空间序列模型，适用于时间序列数据（如EEG信号）。
    这是 CNNMamba 中 SS2D 模块的一维适配版本。
    
    Args:
        d_model: 模型维度
        d_state: SSM状态扩展因子
        d_conv: 卷积核大小
        expand: 模块扩展因子
        dropout: Dropout率
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,  # 使用3，保证兼容因果卷积的宽度限制，同时是奇数保持序列长度
        expand=2.0,
        dropout=0.,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        
        # 一维卷积进行局部混合
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        # 注意力模块
        self.channel_attention = ChannelAttentionModule(self.d_inner)
        self.spatial_attention = SpatialAttentionModule()
        
        # Mamba块
        self.ssm = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner * 2),
            nn.SiLU(),
            nn.Linear(self.d_inner * 2, self.d_inner)
        )
        
        # 标准化与输出投影
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        # 激活函数
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
        Returns:
            输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 分离内容和门控路径
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # [B, T, d_inner], [B, T, d_inner]
        
        # 对门控路径应用通道和空间注意力
        z = z.transpose(1, 2)  # [B, d_inner, T]
        z = self.channel_attention(z) * z
        z = self.spatial_attention(z) * z
        z = z.transpose(1, 2)  # [B, T, d_inner]
        
        # 利用卷积处理内容路径
        x_conv = x.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.act(self.conv1d(x_conv))  # [B, d_inner, T]
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
        
        # 通过SSM处理序列
        y = self.ssm(x_conv)  # [B, T, d_inner]
        
        # 应用门控并归一化
        y = y * F.silu(z)
        y = self.out_norm(y)
        
        # 最终投影
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out 