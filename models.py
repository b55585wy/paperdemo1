import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from hlfae import HLFAE
import logging

class UEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size, middle_layer_filter, depth, activation):
        super(UEncoder, self).__init__()
        layers = []
        current_channels = in_channels
        
        # 修改点1：修正中间层通道数计算逻辑
        for i in range(depth):
            # 首层保持输入通道，后续层使用middle_layer_filter
            actual_in = in_channels if i == 0 else middle_layer_filter
            layers.append(nn.Conv2d(
                in_channels=actual_in,
                out_channels=middle_layer_filter,
                kernel_size=kernel_size,
                padding='same'))
            layers.append(nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid())
        
        # 修改点2：修正最终输出层通道转换
        layers.append(nn.Conv2d(
            in_channels=middle_layer_filter,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same'))
            
        layers.append(nn.MaxPool2d((pooling_size, 1)))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        logging.info(f"UEncoder input shape: {x.shape}")
        output = self.encoder(x)
        logging.info(f"UEncoder output shape: {output.shape} | "
                    f"config: in={self.encoder[0].in_channels}, "
                    f"out={self.encoder[-2].out_channels}, "
                    f"pool={self.encoder[-1].kernel_size}")
        return output

class MSE(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, dilation_sizes=None, activation='relu'):
        super(MSE, self).__init__()
        
        # 处理默认值和检查参数类型
        if dilation_sizes is None:
            dilation = 1
        elif isinstance(dilation_sizes, list):
            dilation = dilation_sizes[0]  # 如果是列表，取第一个元素
        else:
            dilation = dilation_sizes  # 如果是整数，直接使用
            
        self.mse = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(kernel_size, 1),
                dilation=(dilation, 1),  # 直接使用整数
                padding='same'
            ),
            nn.BatchNorm2d(filters),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

    def forward(self, x):
        logging.info(f"MSE input shape: {x.shape} | "
                    f"config: in={self.mse[0].in_channels}, "
                    f"out={self.mse[0].out_channels}, "
                    f"dilation={self.mse[0].dilation}")
        return self.mse(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.upsample(x)

class SingleSalientModel(nn.Module):
    def __init__(self, padding='same', **kwargs):
        super(SingleSalientModel, self).__init__()
        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_len']
        self.sequence_length = kwargs['preprocess']['sequence_epochs']
        self.filters = kwargs['train']['filters']
        self.kernel_size = kwargs['train']['kernel_size']
        self.pooling_sizes = kwargs['train']['pooling_sizes']
        self.dilation_sizes = kwargs['train']['dilation_sizes']
        self.activation = kwargs['train']['activation']
        self.u_depths = kwargs['train']['u_depths']
        self.u_inner_filter = kwargs['train']['u_inner_filter']
        self.mse_filters = kwargs['train']['mse_filters']
        self.hlfae = HLFAE(dim=1)  # 单通道处理
        self.init_model()

    def init_model(self):
        # Encoder
        self.encoder1 = UEncoder(1, self.filters[0], self.kernel_size, self.pooling_sizes[0], 
                               self.u_inner_filter, self.u_depths[0], self.activation)
        self.encoder2 = UEncoder(self.filters[0], self.filters[1], self.kernel_size, 
                               self.pooling_sizes[1], self.u_inner_filter, self.u_depths[1], self.activation)
        self.encoder3 = UEncoder(self.filters[1], self.filters[2], self.kernel_size, 
                               self.pooling_sizes[2], self.u_inner_filter, self.u_depths[2], self.activation)
        self.encoder4 = UEncoder(self.filters[2], self.filters[3], self.kernel_size, 
                               self.pooling_sizes[3], self.u_inner_filter, self.u_depths[3], self.activation)
        self.encoder5 = UEncoder(self.filters[3], self.filters[4], self.kernel_size, 
                               self.pooling_sizes[3], self.u_inner_filter, self.u_depths[3], self.activation)

        # MSE
        self.mse1 = MSE(self.filters[0], self.mse_filters[0], self.kernel_size, self.dilation_sizes[0], self.activation)
        self.mse2 = MSE(self.filters[1], self.mse_filters[1], self.kernel_size, self.dilation_sizes[1], self.activation)
        self.mse3 = MSE(self.filters[2], self.mse_filters[2], self.kernel_size, self.dilation_sizes[2], self.activation)
        self.mse4 = MSE(self.filters[3], self.mse_filters[3], self.kernel_size, self.dilation_sizes[3], self.activation)
        self.mse5 = MSE(self.filters[4], self.mse_filters[4], self.kernel_size, self.dilation_sizes[-1], self.activation)

        # Decoder
        self.upsample4 = Upsample(scale_factor=(self.pooling_sizes[3], 1))
        self.decoder4 = UEncoder(self.filters[3] * 2, self.filters[3], self.kernel_size, 
                               self.pooling_sizes[3], self.u_inner_filter, self.u_depths[3], self.activation)
        self.upsample3 = Upsample(scale_factor=(self.pooling_sizes[2], 1))
        self.decoder3 = UEncoder(self.filters[2] * 2, self.filters[2], self.kernel_size, 
                               self.pooling_sizes[2], self.u_inner_filter, self.u_depths[2], self.activation)
        self.upsample2 = Upsample(scale_factor=(self.pooling_sizes[1], 1))
        self.decoder2 = UEncoder(self.filters[1] * 2, self.filters[1], self.kernel_size, 
                               self.pooling_sizes[1], self.u_inner_filter, self.u_depths[1], self.activation)
        self.upsample1 = Upsample(scale_factor=(self.pooling_sizes[0], 1))
        self.decoder1 = UEncoder(self.filters[0] * 2, self.filters[0], self.kernel_size, 
                               self.pooling_sizes[0], self.u_inner_filter, self.u_depths[0], self.activation)

        # Output layers
        self.zero_pad = nn.ZeroPad2d((0, 0, self.sequence_length * self.sleep_epoch_length // 2, 
                                     self.sequence_length * self.sleep_epoch_length // 2))
        self.reshape_conv = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=1)
        self.final_conv = nn.Conv2d(self.filters[0], 5, kernel_size=(self.kernel_size, 1), padding='same')

    def preprocess_input(self, x):
        """
        预处理输入数据使其符合模型要求的维度
        输入 x: [channel=3, 大组数, 大组长度, 30s数据基本单元, h, n]
        输出: [batch_size, channels=1, height, width=1]
        """
        # 调整维度顺序，将channel移到第二维
        x = x.permute(1, 0, 2, 3, 4, 5)  # [大组数, channel=3, 大组长度, 30s数据, h, n]
        
        # 计算batch_size和height
        batch_size = x.shape[0]  # 大组数作为batch_size
        height = x.shape[2] * x.shape[3]  # 大组长度 * 30s数据长度
        
        # 重新整形，只保留第一个通道
        x = x[:, 0:1, :, :, :, :]  # 只取第一个通道
        x = x.reshape(batch_size, 1, height, -1)  # 合并剩余维度到最后一维
        x = x.squeeze(-1)  # 移除最后一维
        x = x.unsqueeze(-1)  # 添加宽度维度为1
        
        logging.info(f"Preprocessed input shape: {x.shape}")
        return x
    
    def forward(self, x):
        # 1. 预处理输入数据
        x = self.preprocess_input(x)
        logging.info(f"After preprocess shape: {x.shape}")

        # 2. HLFAE 处理
        x = self.hlfae(x)
        logging.info(f"After HLFAE shape: {x.shape}")

        # 3. UUNet 处理
        u1 = self.encoder1(x)
        logging.info(f"After encoder1 shape: {u1.shape}")
        u1 = self.mse1(u1)
        logging.info(f"After mse1 shape: {u1.shape}")
        pool1 = F.max_pool2d(u1, kernel_size=(self.pooling_sizes[0], 1))

        u2 = self.encoder2(pool1)
        u2 = self.mse2(u2)
        pool2 = F.max_pool2d(u2, kernel_size=(self.pooling_sizes[1], 1))

        u3 = self.encoder3(pool2)
        u3 = self.mse3(u3)
        pool3 = F.max_pool2d(u3, kernel_size=(self.pooling_sizes[2], 1))

        u4 = self.encoder4(pool3)
        u4 = self.mse4(u4)
        pool4 = F.max_pool2d(u4, kernel_size=(self.pooling_sizes[3], 1))

        u5 = self.encoder5(pool4)
        u5 = self.mse5(u5)

        # Decoder
        up4 = self.upsample4(u5)
        d4 = self.decoder4(torch.cat([up4, u4], dim=1))

        up3 = self.upsample3(d4)
        d3 = self.decoder3(torch.cat([up3, u3], dim=1))

        up2 = self.upsample2(d3)
        d2 = self.decoder2(torch.cat([up2, u2], dim=1))

        up1 = self.upsample1(d2)
        d1 = self.decoder1(torch.cat([up1, u1], dim=1))

        # Output
        zpad = self.zero_pad(d1)
        reshape = self.reshape_conv(zpad)
        pool = F.avg_pool2d(reshape, kernel_size=(1, self.sleep_epoch_length))
        out = self.final_conv(pool)
        return F.softmax(out, dim=1)

class TwoStreamSalientModel(nn.Module):
    def __init__(self, padding='same', **kwargs):
        super(TwoStreamSalientModel, self).__init__()
        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_len']
        self.sequence_length = kwargs['preprocess']['sequence_epochs']
        # 检查 'train' 字典是否存在 'filters' 键
        if 'train' in kwargs and 'filters' in kwargs['train']:
            self.filters = kwargs['train']['filters']
        else:
            # 可以根据实际情况设置默认值
            self.filters = [16, 32, 64, 128, 256]
        self.kernel_size = kwargs['train']['kernel_size']
        self.pooling_sizes = kwargs['train']['pooling_sizes']
        self.dilation_sizes = kwargs['train']['dilation_sizes']
        # 修改这里，直接从kwargs中获取activation
        self.activation = kwargs.get('activation', 'relu')  # 若没有提供，默认使用'relu'
        self.u_depths = kwargs['train']['u_depths']
        self.u_inner_filter = kwargs['train']['u_inner_filter']
        self.mse_filters = kwargs['train']['mse_filters']

        self.eeg_branch = self.build_branch()
        self.eog_branch = self.build_branch()

        # Merge layers
        # 修改这里的合并层定义
        # self.merge_multiply = nn.Multiply()  # 删除这行
        # self.merge_add = nn.Add()  # 删除这行
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_dense1 = nn.Linear(self.filters[0], self.filters[0] // 4)
        self.se_dense2 = nn.Linear(self.filters[0] // 4, self.filters[0])
        self.se_sigmoid = nn.Sigmoid()
        self.reshape_conv = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=1)
        self.final_conv = nn.Conv2d(self.filters[0], 5, kernel_size=(self.kernel_size, 1), padding='same')

    # 修改编码器定义部分
    def build_branch(self):
        # 修正编码器定义（保持通道连续性）
        encoder1 = UEncoder(
            in_channels=1,
            out_channels=self.mse_filters[0],
            kernel_size=self.kernel_size,
            pooling_size=self.pooling_sizes[0],
            middle_layer_filter=self.u_inner_filter,
            depth=self.u_depths[0],
            activation=self.activation
        )
        
        encoder2 = UEncoder(
            in_channels=self.mse_filters[0],  # 确保输入通道与前一层的输出一致
            out_channels=self.mse_filters[1],
            kernel_size=self.kernel_size, 
            pooling_size=self.pooling_sizes[1],
            middle_layer_filter=self.u_inner_filter,
            depth=self.u_depths[1],
            activation=self.activation
        )
    
        # 新增中间层通道校验
        logging.info(f"Encoder1 output channels: {self.mse_filters[0]}")
        logging.info(f"Encoder2 input channels: {self.mse_filters[0]}")
        encoder3 = UEncoder(
            in_channels=self.mse_filters[1],
            out_channels=self.mse_filters[2],
            kernel_size=self.kernel_size,
            pooling_size=self.pooling_sizes[2],
            middle_layer_filter=self.u_inner_filter,
            depth=self.u_depths[2],
            activation=self.activation
        )
    
        encoder4 = UEncoder(
            in_channels=self.mse_filters[2],
            out_channels=self.mse_filters[3],
            kernel_size=self.kernel_size,
            pooling_size=self.pooling_sizes[3],
            middle_layer_filter=self.u_inner_filter,
            depth=self.u_depths[3],
            activation=self.activation
        )
    
        # 修改点3：调整encoder5的输入通道定义
        encoder5 = UEncoder(
            in_channels=self.mse_filters[3],  # 确保输入通道与encoder4输出一致
            out_channels=self.mse_filters[4],  # 256
            kernel_size=self.kernel_size,
            pooling_size=self.pooling_sizes[3],
            middle_layer_filter=self.u_inner_filter,  # 256
            depth=self.u_depths[3],  # 4
            activation=self.activation
        )
    
        # MSE部分保持不变
        # 修正MSE层的参数传递
        mse1 = MSE(self.mse_filters[0], self.mse_filters[0], self.kernel_size, self.dilation_sizes[0], self.activation)
        mse2 = MSE(self.mse_filters[1], self.mse_filters[1], self.kernel_size, self.dilation_sizes[1], self.activation)
        mse3 = MSE(self.mse_filters[2], self.mse_filters[2], self.kernel_size, self.dilation_sizes[2], self.activation)
        mse4 = MSE(self.mse_filters[3], self.mse_filters[3], self.kernel_size, self.dilation_sizes[3], self.activation)
        mse5 = MSE(self.mse_filters[4], self.mse_filters[4], self.kernel_size, self.dilation_sizes[-1], self.activation)
        
        # 修改decoder定义
        upsample4 = Upsample(scale_factor=(self.pooling_sizes[3], 1))
        decoder4 = UEncoder(
            self.mse_filters[4],  # Correct input channels
            self.mse_filters[3],
            self.kernel_size,
            self.pooling_sizes[3],
            self.u_inner_filter,
            self.u_depths[3],
            self.activation
        )
    
        upsample3 = Upsample(scale_factor=(self.pooling_sizes[2], 1))
        decoder3 = UEncoder(
            self.mse_filters[3],  # Correct input channels
            self.mse_filters[2],
            self.kernel_size,
            self.pooling_sizes[2],
            self.u_inner_filter,
            self.u_depths[2],
            self.activation
        )
    
        upsample2 = Upsample(scale_factor=(self.pooling_sizes[1], 1))
        decoder2 = UEncoder(
            self.mse_filters[2],  # Correct input channels
            self.mse_filters[1],
            self.kernel_size,
            self.pooling_sizes[1],
            self.u_inner_filter,
            self.u_depths[1],
            self.activation
        )
    
        upsample1 = Upsample(scale_factor=(self.pooling_sizes[0], 1))
        decoder1 = UEncoder(
            self.mse_filters[1],  # Correct input channels
            self.mse_filters[0],
            self.kernel_size,
            self.pooling_sizes[0],
            self.u_inner_filter,
            self.u_depths[0],
            self.activation
        )
    
        # Zero padding
        zero_pad = nn.ZeroPad2d((0, 0, self.sequence_length * self.sleep_epoch_length // 2, self.sequence_length * self.sleep_epoch_length // 2))
    
        # Now create branch_layers list with all defined components
        branch_layers = [
            encoder1, mse1,
            encoder2, mse2,
            encoder3, mse3,
            encoder4, mse4,
            encoder5, mse5,
            upsample4, decoder4,
            upsample3, decoder3,
            upsample2, decoder2,
            upsample1, decoder1,
            zero_pad,
            nn.Conv2d(self.filters[0], self.filters[0], kernel_size=(3, 3), padding='same'),
            nn.Conv2d(self.filters[0], self.filters[0], kernel_size=(3, 3), padding='same')
        ]
    
        return nn.Sequential(*branch_layers)

    def preprocess_input(self, x):
        """
        预处理双流输入数据
        输入 x: tuple of (eeg_data, eog_data)
        输出: 两个形状为 [batch_size, channels=1, height, width=1] 的张量
        """
        eeg_data, eog_data = x
        
        def process_single_stream(data):
            logging.info(f"Input shape before processing: {data.shape}")
            if len(data.shape) == 5:
                batch_size, seq_len, epoch_len, h, w = data.shape
                # 增加维度校验
                assert w == 1, f"Unexpected width dimension: {w}"
                data = data.view(batch_size, 1, seq_len * epoch_len * h, 1)  # 合并h维度
                logging.info(f"Reshaped 5D input to: {data.shape}")
            return data
        
        eeg_processed = process_single_stream(eeg_data)
        eog_processed = process_single_stream(eog_data)
        
        return eeg_processed, eog_processed

    def forward(self, x):
        # 分离输入
        eeg_input, eog_input = self.preprocess_input(x)
        logging.info(f"EEG input shape: {eeg_input.shape}")
        logging.info(f"EOG input shape: {eog_input.shape}")
        
        # 确保输入维度正确
        if eeg_input.shape[1] != 1:
            logging.warning(f"Unexpected EEG input channels: {eeg_input.shape[1]}, reshaping to 1 channel")
            eeg_input = eeg_input[:, :1, :, :]  # 只保留第一个通道
        
        if eog_input.shape[1] != 1:
            logging.warning(f"Unexpected EOG input channels: {eog_input.shape[1]}, reshaping to 1 channel")
            eog_input = eog_input[:, :1, :, :]  # 只保留第一个通道
        
        # EEG 分支
        eeg_output = self.eeg_branch(eeg_input)
        logging.info(f"EEG output shape: {eeg_output.shape}")
        
        # EOG 分支
        eog_output = self.eog_branch(eog_input)
        logging.info(f"EOG output shape: {eog_output.shape}")
    
        # 合并分支
        multiply = eeg_output * eog_output
        merge = eeg_output + eog_output + multiply
    
        # Squeeze-and-Excitation (SE) 块
        se = self.global_avg_pool(merge)
        se = se.view(se.size(0), -1)
        se = F.relu(self.se_dense1(se))
        se = self.se_sigmoid(self.se_dense2(se))
        se = se.view(se.size(0), self.filters[0], 1, 1)
        reweighted = merge * se
    
        # 最终层
        reshape = self.reshape_conv(reweighted)
        
        # 添加全局平均池化，将时间维度压缩
        global_pool = F.adaptive_avg_pool2d(reshape, (1, 1))
        
        # 最终卷积层
        out = self.final_conv(global_pool)
        
        # 重塑输出以匹配目标标签的形状
        out = out.squeeze(-1).squeeze(-1)  # 移除最后两个维度
        
        # 确保输出维度与标签匹配
        batch_size = out.shape[0]
        seq_len = 20  # 从配置中的 sequence_epochs
        
        # 重塑为 [batch, seq, classes]
        out = out.unsqueeze(1).repeat(1, seq_len, 1)
        
        return out

if __name__ == "__main__":
    import yaml
    import logging
    
    # 设置详细的日志记录
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    with open("hyperparameters.yaml", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    # 创建更真实的测试数据
    batch_size = 2
    sequence_length = hyp["preprocess"]["sequence_epochs"]
    sleep_epoch_length = hyp["sleep_epoch_len"]
    
    # 创建5D测试数据
    eeg_input = torch.randn(batch_size, sequence_length, sleep_epoch_length, 1, 1)
    eog_input = torch.randn(batch_size, sequence_length, sleep_epoch_length, 1, 1)
    
    logging.info(f"Test EEG input shape: {eeg_input.shape}")
    logging.info(f"Test EOG input shape: {eog_input.shape}")

    # 初始化模型
    model = TwoStreamSalientModel(**hyp)

    # 测试前向传播
    try:
        output = model((eeg_input, eog_input))
        logging.info(f"Output shape: {output.shape}")
    except Exception as e:
        logging.error(f"Forward pass failed: {str(e)}")
        logging.error(f"Error type: {type(e)}")
