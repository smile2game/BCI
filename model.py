'''
Coding: utf-8
Author: vector-wlc
Date: 2022-12-30 20:53:04
Description:
'''

import torch
import torch.nn
import numpy as np


class InputBlockTime(torch.nn.Module):
    def __init__(self, time_channels, time_size, spatial_channels, spatial_size, pool_param=(1, 2), dropout=0.5):
        super(InputBlockTime, self).__init__()

        self.block = torch.nn.Sequential(
            # time filter
            torch.nn.Conv2d(in_channels=1, out_channels=time_channels,
                            kernel_size=time_size, padding=(0, time_size[1] // 2), bias=False),


            # spatial filter
            torch.nn.Conv2d(in_channels=time_channels, groups=time_channels,
                            out_channels=spatial_channels, kernel_size=spatial_size),

            torch.nn.BatchNorm2d(spatial_channels),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=pool_param,
                               stride=pool_param),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class FeatureBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_param=(1, 15), pool_param=(1, 2), dropout=0.5, padding=(-1, -1, -1, -1)):
        super(FeatureBlock, self).__init__()
        if (padding[0] == -1):
            padding = (conv_param[1] // 2, conv_param[1] // 2, 0, 0)

        self.padding = torch.nn.ZeroPad2d(padding=padding)

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_param,
            bias=False,
        )

        self.norm = torch.nn.BatchNorm2d(
            num_features=out_channels
        )

        self.elu = torch.nn.ELU()

        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_param,
            stride=pool_param
        )

        self.drop_out = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.elu(x)
        x = self.pool(x)
        return self.drop_out(x)


class DeepConvNet(torch.nn.Module):
    def __init__(self, sample, class_num, time_channels, time_size,
                 spatial_channels, spatial_size,
                 feature_pool_size, feature_channels_list, dropout):
        super(DeepConvNet, self).__init__()

        self.input_block = InputBlockTime(time_channels=time_channels, time_size=time_size,
                                          spatial_channels=spatial_channels, spatial_size=spatial_size,
                                          pool_param=feature_pool_size, dropout=dropout)

        self.feature_block_list = torch.nn.Sequential()

        pre_channels = spatial_channels

        for channel in feature_channels_list:
            self.feature_block_list.add_module(
                f"feature {channel}",
                FeatureBlock(in_channels=pre_channels, out_channels=channel,
                             pool_param=feature_pool_size, dropout=dropout)
            )
            pre_channels = channel

        # 造一个数据来得到全连接层之前数据的 shape
        # 这样就不用手动计算数据的 shape 了，是一个实用技巧
        tmp_data = torch.Tensor(np.ones((1, 1, 64, sample), dtype=float))
        tmp_data = self.input_block(tmp_data)
        tmp_data = self.feature_block_list(tmp_data)
        tmp_data = tmp_data.view(tmp_data.size(0), -1)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(tmp_data.shape[1],
                            class_num)
        )

    def forward(self, x):
        out2 = self.input_block(x)
        x = self.feature_block_list(out2)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return torch.nn.functional.softmax(x, dim=1)

if __name__ == '__main__':
    MyNet= DeepConvNet(
        sample=256, class_num=2,
        time_channels=25, time_size=(1, 9),
        spatial_channels=50, spatial_size=(64, 1),
        feature_pool_size=(1, 3), feature_channels_list=[100, 200], dropout=0.5)
    print(MyNet)