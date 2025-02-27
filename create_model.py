import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

def upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    使用双线性插值进行上采样
    """
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

class BNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, padding: str = 'same', activation: str = 'relu'):
        super().__init__()
        if padding == 'same':
            padding = ((kernel_size - 1) * dilation) // 2
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                             dilation=(dilation, dilation), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class UEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 pooling_size: int, middle_layer_filter: int, depth: int,
                 padding: str = 'same', activation: str = 'relu'):
        super().__init__()
        
        self.depth = depth
        self.initial_conv = BNConv(in_channels, out_channels, kernel_size, 
                                 padding=padding, activation=activation)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        for d in range(depth - 1):
            self.encoder_layers.append(
                BNConv(out_channels if d == 0 else middle_layer_filter,
                       middle_layer_filter, kernel_size, padding=padding,
                       activation=activation)
            )
            if d != depth - 2:
                self.pools.append(nn.MaxPool2d((pooling_size, 1)))
                
        # 中间层
        self.middle = BNConv(middle_layer_filter, middle_layer_filter,
                            kernel_size, padding=padding, activation=activation)
        
        # 解码器层
        self.decoder_layers = nn.ModuleList()
        for d in range(depth - 1, 0, -1):
            ch = out_channels if d == 1 else middle_layer_filter
            self.decoder_layers.append(
                BNConv(middle_layer_filter * 2, ch, kernel_size,
                       padding=padding, activation=activation)
            )

    def forward(self, x):
        # 初始卷积
        x0 = self.initial_conv(x)
        x = x0
        
        # 编码过程
        skip_connections = []
        for i, encoder in enumerate(self.encoder_layers):
            x = encoder(x)
            skip_connections.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
                
        # 中间层处理
        x = self.middle(x)
        
        # 解码过程
        for i, decoder in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            x = upsample(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            
        return x + x0

class MSE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation_rates: List[int], padding: str = 'same',
                 activation: str = 'relu'):
        super().__init__()
        
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList([
            BNConv(in_channels, out_channels, kernel_size,
                  dilation=dr, padding=padding, activation=activation)
            for dr in dilation_rates
        ])
        
        # 下采样卷积
        self.down_conv1 = nn.Conv2d(out_channels * len(dilation_rates),
                                   out_channels * 2, (kernel_size, 1),
                                   padding='same' if padding == 'same' else 0)
        self.down_conv2 = nn.Conv2d(out_channels * 2, out_channels,
                                   (kernel_size, 1),
                                   padding='same' if padding == 'same' else 0)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 多尺度特征提取
        multi_scale_features = [conv(x) for conv in self.conv_layers]
        x = torch.cat(multi_scale_features, dim=1)
        
        # 特征融合
        x = self.activation(self.down_conv1(x))
        x = self.activation(self.down_conv2(x))
        return self.bn(x)