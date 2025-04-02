import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

# 特征提取模块
class feature_extraction(nn.Module):
    def __init__(self, in_channels):
        super(feature_extraction, self).__init__()
        # 将Conv2d改为Conv1d
        self.dilate1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.dilate2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.dilate3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.dilate4 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, dilation=7, padding=7),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.convmlp = nn.Sequential(
            nn.Conv1d(4 * in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 计算四种不同膨胀卷积的输出
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(x)
        dilate3_out = self.dilate3(x)
        dilate4_out = self.dilate4(x)
        # 拼接四个不同膨胀卷积的输出
        cnn_out = torch.cat([dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)
        # 使用1x1卷积进行通道压缩
        out = self.convmlp(cnn_out)
        return out

# 结合卷积、批归一化与ReLU激活的模块
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm1d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)  # 卷积+批归一化
        if self.act == True:  # 如果设置了激活，则进行ReLU激活
            x = self.relu(x)
        return x

# Squeeze-and-Excitation (SE) 注意力机制模块
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 改为1d池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    # 权重初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 初始化卷积层权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 初始化BatchNorm权重
                init.constant_(m.bias, 0)  # 初始化偏置
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 初始化全连接层权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置

    def forward(self, x):
        b, c, _ = x.size()  # 修改size获取
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# 高低频特征自适应增强模块
class HLFAE(nn.Module):  # HLFAE高低频特征自适应增强模块
    def __init__(self, dim, ratio=16):
        super(HLFAE, self).__init__()

        self.down = nn.AvgPool1d(kernel_size=2)  # 改为1d池化
        self.high_feature = feature_extraction(dim)  # 高频特征提取模块
        self.low = feature_extraction(dim)  # 低频特征提取模块
        self.conv1 = CBR(2 * dim, dim)  # 结合高频和低频特征的卷积模块

        # 用于计算低频特征的通道注意力
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 通过全连接层生成注意力权重
        self.fc1 = nn.Conv1d(dim, dim // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(dim // ratio, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid生成注意力权重
        self.att = SEAttention(dim)  # SE注意力模块

    def forward(self, x):
        # 检查输入维度
        assert len(x.shape) == 4, f"Expected 4D input tensor, got shape {x.shape}"
        assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"
        assert x.shape[3] == 1, f"Expected width 1, got {x.shape[3]}"
        
        # 第一步：分解高频和低频特征
        low = self.down(x)  # 低频特征通过池化获得
        high = x - F.interpolate(low, size=x.size()[-1], mode='linear')
        high = self.high_feature(high)  # 提取高频特征

        # 第二步：低频特征的增强
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(low))))  # 平均池化后经过FC生成注意力权重
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(low))))  # 最大池化后经过FC生成注意力权重
        low = low * self.sigmoid(avg_out + max_out)  # 低频特征加权
        low = F.interpolate(low, size=x.size()[-1], mode='linear')  # 恢复原始尺寸

        # 第三步：融合高低频特征
        out = torch.cat([high, low], dim=1)  # 高低频特征拼接
        out = self.att(self.conv1(out)) + x  # 使用SE注意力模块处理融合后的特征，并加回输入特征
        return out

# 测试代码：输入BCHW，输出BCHW
if __name__ == '__main__':
    # 实例化模型对象
    model = HLFAE(dim=32)
    # 生成一维信号数据，形状为(batch_size, channels, signal_length)
    input = torch.randn(1, 32, 64)
    # 执行前向传播
    output = model(input)
    # 打印输入和输出的尺寸
    print('input_size:', input.size())
    print('output_size:', output.size())