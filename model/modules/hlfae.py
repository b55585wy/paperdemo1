import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class FeatureExtraction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 修改卷积核大小和padding以适应睡眠数据的一维特性
        self.dilate1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), dilation=1, padding=(0, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.dilate2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), dilation=3, padding=(0, 3)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.dilate3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), dilation=5, padding=(0, 5)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.dilate4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), dilation=7, padding=(0, 7)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.convmlp = nn.Sequential(
            nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(2 * in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(x)
        dilate3_out = self.dilate3(x)
        dilate4_out = self.dilate4(x)
        out = torch.cat([dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)
        return self.convmlp(out)

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        # 修改为1D卷积操作
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(1, kernel_size), 
                     padding=(0, padding), dilation=(1, dilation), 
                     bias=False, stride=(1, stride)),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HLFAE(nn.Module):
    def __init__(self, dim, ratio=16):
        super().__init__()
        # 修改池化操作以适应睡眠数据的一维特性
        self.down = nn.AvgPool2d(kernel_size=(1, 2))
        self.high_feature = FeatureExtraction(dim)
        self.low = FeatureExtraction(dim)
        self.conv1 = CBR(2 * dim, dim)
        
        # 添加通道注意力机制
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # 保持时间维度
        self.max_pool = nn.AdaptiveMaxPool2d((1, None))
        
        self.fc1 = nn.Conv2d(dim, dim // ratio, (1, 1), bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(dim // ratio, dim, (1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.att = SEAttention(dim)

    def forward(self, x):
        # 分解高低频特征
        low = self.down(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='nearest')  # 改用nearest插值
        high = self.high_feature(high)
        
        # 低频特征增强。这里原始代码做了全局池化层和平均池化层
        low = self.low(low)
        low = F.interpolate(low, size=x.size()[-2:], mode='nearest')
        
        # 特征融合
        out = torch.cat([high, low], dim=1)
        out = self.att(self.conv1(out)) + x
        return out

if __name__ == '__main__':
    # 测试代码
    model = HLFAE(dim=1)  # 改为1，因为每个通道单独处理
    # 生成与预处理后数据形状相匹配的测试数据
    # [batch_size, channels(1), sequence_length, 1]
    x = torch.randn(4, 1, 3000, 1)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")