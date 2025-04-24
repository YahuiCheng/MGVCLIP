import torch
import torch.nn as nn


def harmonic_average(x, dim=1):
    epsilon = 1e-8
    x = 1 / (x + epsilon)
    x = torch.mean(x, dim=dim, keepdim=True)
    x = 1 / (x + epsilon)
    return x


class Conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p):
        super().__init__()

        self.channel_in = in_channel
        self.channel_out = out_channel
        self.group = self.channel_in
        self.kernel_size = k
        self.stride = s
        self.pad = p

        self.depth_conv = nn.Conv2d(self.channel_in, self.channel_in, self.kernel_size, self.stride, self.pad, groups=self.group, bias=False)
        self.point_conv = nn.Conv2d(self.channel_in, self.channel_out, 1, bias=False)

        self.bn = nn.BatchNorm2d(self.channel_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Multi_scale_adapter(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_cchannel = in_channel
        self.out_chanlle = out_channel

        self.conv_3 = Conv_block(self.in_cchannel, self.out_chanlle, 3, 1, 1)
        self.conv_5 = Conv_block(self.in_cchannel, self.out_chanlle, 5, 1, 2)
        self.conv_7 = Conv_block(self.in_cchannel, self.out_chanlle, 7, 1, 3)
        self.conv = Conv_block(self.in_cchannel, self.out_chanlle, 3, 1, 1)

        self.fc1 = nn.Linear(225, 112)
        self.act_fc = nn.GELU()
        self.dp = nn.Dropout(0.1)
        self.fc2 = nn.Linear(112, 225)
        self.initial_paras()

    def initial_paras(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        harmonic_feature = x
        harmonic_feature = harmonic_average(harmonic_feature, dim=1)
        N, C, H, W = harmonic_feature.shape
        harmonic_feature = torch.flatten(harmonic_feature, 1)
        harmonic_feature = self.fc1(harmonic_feature)
        harmonic_feature = self.act_fc(harmonic_feature)
        harmonic_feature = self.dp(harmonic_feature)
        harmonic_feature = self.fc2(harmonic_feature)
        harmonic_feature_gate = torch.sigmoid(harmonic_feature)
        harmonic_feature_gate = harmonic_feature_gate.view(N, C, H, W)

        x_3 = self.conv_3(x)
        x_5 = self.conv_5(x)
        x_7 = self.conv_7(x)

        x = x_3 + x_5 + x_7
        x = self.conv(x)
        x = harmonic_feature_gate * x
        return x
