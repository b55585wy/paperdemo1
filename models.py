import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from modules.hlfae import HLFAE

class UEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size, middle_layer_filter, depth, activation):
        super(UEncoder, self).__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(in_channels if i == 0 else middle_layer_filter, middle_layer_filter, kernel_size, padding='same'))
            layers.append(nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid())
        layers.append(nn.Conv2d(middle_layer_filter, out_channels, kernel_size, padding='same'))
        layers.append(nn.MaxPool2d((pooling_size, 1)))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class MSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates, activation):
        super(MSE, self).__init__()
        layers = []
        for dilation in dilation_rates:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid())
        self.mse = nn.Sequential(*layers)

    def forward(self, x):
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
        self.mse1 = MSE(self.filters[0], self.mse_filters[0], self.kernel_size, self.dilation_sizes, self.activation)
        self.mse2 = MSE(self.filters[1], self.mse_filters[1], self.kernel_size, self.dilation_sizes, self.activation)
        self.mse3 = MSE(self.filters[2], self.mse_filters[2], self.kernel_size, self.dilation_sizes, self.activation)
        self.mse4 = MSE(self.filters[3], self.mse_filters[3], self.kernel_size, self.dilation_sizes, self.activation)
        self.mse5 = MSE(self.filters[4], self.mse_filters[4], self.kernel_size, self.dilation_sizes, self.activation)

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
    def forward(self, x):
        # 首先通过HLFAE进行特征增强
        x = self.hlfae(x)
        # 原有的编码器-解码器流程
        # Encoder
        u1 = self.encoder1(x)
        u1 = self.mse1(u1)
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
        self.filters = kwargs['train']['filters']
        self.kernel_size = kwargs['train']['kernel_size']
        self.pooling_sizes = kwargs['train']['pooling_sizes']
        self.dilation_sizes = kwargs['train']['dilation_sizes']
        self.activation = kwargs['train']['activation']
        self.u_depths = kwargs['train']['u_depths']
        self.u_inner_filter = kwargs['train']['u_inner_filter']
        self.mse_filters = kwargs['train']['mse_filters']

        self.eeg_branch = self.build_branch()
        self.eog_branch = self.build_branch()

        # Merge layers
        self.merge_multiply = nn.Multiply()
        self.merge_add = nn.Add()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_dense1 = nn.Linear(self.filters[0], self.filters[0] // 4)
        self.se_dense2 = nn.Linear(self.filters[0] // 4, self.filters[0])
        self.se_sigmoid = nn.Sigmoid()
        self.reshape_conv = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=1)
        self.final_conv = nn.Conv2d(self.filters[0], 5, kernel_size=(self.kernel_size, 1), padding='same')

    def build_branch(self):
        # Encoder
        encoder1 = UEncoder(1, self.filters[0], self.kernel_size, self.pooling_sizes[0], self.u_inner_filter, self.u_depths[0], self.activation)
        encoder2 = UEncoder(self.filters[0], self.filters[1], self.kernel_size, self.pooling_sizes[1], self.u_inner_filter, self.u_depths[1], self.activation)
        encoder3 = UEncoder(self.filters[1], self.filters[2], self.kernel_size, self.pooling_sizes[2], self.u_inner_filter, self.u_depths[2], self.activation)
        encoder4 = UEncoder(self.filters[2], self.filters[3], self.kernel_size, self.pooling_sizes[3], self.u_inner_filter, self.u_depths[3], self.activation)
        encoder5 = UEncoder(self.filters[3], self.filters[4], self.kernel_size, self.pooling_sizes[3], self.u_inner_filter, self.u_depths[3], self.activation)

        # MSE
        mse1 = MSE(self.filters[0], self.mse_filters[0], self.kernel_size, self.dilation_sizes, self.activation)
        mse2 = MSE(self.filters[1], self.mse_filters[1], self.kernel_size, self.dilation_sizes, self.activation)
        mse3 = MSE(self.filters[2], self.mse_filters[2], self.kernel_size, self.dilation_sizes, self.activation)
        mse4 = MSE(self.filters[3], self.mse_filters[3], self.kernel_size, self.dilation_sizes, self.activation)
        mse5 = MSE(self.filters[4], self.mse_filters[4], self.kernel_size, self.dilation_sizes, self.activation)

        # Decoder
        upsample4 = Upsample(scale_factor=(self.pooling_sizes[3], 1))
        decoder4 = UEncoder(self.filters[3] * 2, self.filters[3], self.kernel_size, self.pooling_sizes[3], self.u_inner_filter, self.u_depths[3], self.activation)
        upsample3 = Upsample(scale_factor=(self.pooling_sizes[2], 1))
        decoder3 = UEncoder(self.filters[2] * 2, self.filters[2], self.kernel_size, self.pooling_sizes[2], self.u_inner_filter, self.u_depths[2], self.activation)
        upsample2 = Upsample(scale_factor=(self.pooling_sizes[1], 1))
        decoder2 = UEncoder(self.filters[1] * 2, self.filters[1], self.kernel_size, self.pooling_sizes[1], self.u_inner_filter, self.u_depths[1], self.activation)
        upsample1 = Upsample(scale_factor=(self.pooling_sizes[0], 1))
        decoder1 = UEncoder(self.filters[0] * 2, self.filters[0], self.kernel_size, self.pooling_sizes[0], self.u_inner_filter, self.u_depths[0], self.activation)

        # Zero padding
        zero_pad = nn.ZeroPad2d((0, 0, self.sequence_length * self.sleep_epoch_length // 2, self.sequence_length * self.sleep_epoch_length // 2))

        return nn.Sequential(
            encoder1, mse1, nn.MaxPool2d((self.pooling_sizes[0], 1)),
            encoder2, mse2, nn.MaxPool2d((self.pooling_sizes[1], 1)),
            encoder3, mse3, nn.MaxPool2d((self.pooling_sizes[2], 1)),
            encoder4, mse4, nn.MaxPool2d((self.pooling_sizes[3], 1)),
            encoder5, mse5,
            upsample4, decoder4,
            upsample3, decoder3,
            upsample2, decoder2,
            upsample1, decoder1,
            zero_pad
        )

    def forward(self, x):
        eeg_input, eog_input = x
        eeg_output = self.eeg_branch(eeg_input)
        eog_output = self.eog_branch(eog_input)

        # Merge branches
        multiply = self.merge_multiply(eeg_output, eog_output)
        merge = self.merge_add(eeg_output, eog_output, multiply)

        # Squeeze-and-Excitation (SE) block
        se = self.global_avg_pool(merge)
        se = se.view(se.size(0), -1)  # Flatten
        se = F.relu(self.se_dense1(se))
        se = self.se_sigmoid(self.se_dense2(se))
        se = se.view(se.size(0), self.filters[0], 1, 1)  # Reshape back to 4D
        reweighted = merge * se

        # Final layers
        reshape = self.reshape_conv(reweighted)
        pool = F.avg_pool2d(reshape, kernel_size=(1, self.sleep_epoch_length))
        out = self.final_conv(pool)
        return F.softmax(out, dim=1)
if __name__ == "__main__":
    import yaml

    with open("hyperparameters.yaml", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    # Create a dummy input tensor
    batch_size = 2
    sequence_length = hyp["preprocess"]["sequence_epochs"]
    sleep_epoch_length = hyp["sleep_epoch_len"]
    input_shape = (sequence_length * sleep_epoch_length, 1, 1)
    eeg_input = torch.randn(batch_size, *input_shape)
    eog_input = torch.randn(batch_size, *input_shape)

    # Initialize the model
    model = TwoStreamSalientModel(**hyp)

    # Forward pass
    output = model((eeg_input, eog_input))
    print("Output shape:", output.shape)
