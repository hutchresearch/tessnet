import torch
from torch import nn


class LSTM3d(nn.Module):
    def __init__(self,
                 classifier_head=False,
                 period_head=False,
                 num_classes=4,
                 re_zero=False
                 ):
        super().__init__()

        self.classifier_head = classifier_head
        self.period_head = period_head
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            ConvBlock(1, 4, 8, re_zero=re_zero),
            IdentityBlock(8, 8, 8, re_zero=re_zero),
            IdentityBlock(8, 8, 8, re_zero=re_zero)
        )

        self.layer2 = nn.Sequential(
            ConvBlock(8, 16, 32, re_zero=re_zero),
            IdentityBlock(32, 32, 32, re_zero=re_zero),
            IdentityBlock(32, 32, 32, re_zero=re_zero)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        if self.period_head:
            self.period_fc = nn.Linear(128, 1)
        if self.classifier_head:
            self.classifier_fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = torch.flatten(x, start_dim=2)

        _, (h_n, _) = self.lstm(x)
        enc = h_n[-1]

        if self.classifier_head:
            class_pred = self.classifier_fc(enc)
        else:
            class_pred = torch.zeros((batch_size, self.num_classes)).to(x.device)

        if self.period_head:
            period_pred = self.period_fc(enc)
        else:
            period_pred = torch.zeros((batch_size, 1)).to(x.device)

        return class_pred, period_pred


class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, re_zero=False):
        super().__init__()
        self.re_zero = re_zero
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=3)
        self.downsample_bn = nn.BatchNorm3d(out_channels)
        if self.re_zero:
            self.gating_param = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.gating_param, 0.1)
            nn.init.zeros_(self.downsample.weight)

        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_channels)
        self.conv3 = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        res = self.downsample(identity)
        res = self.downsample_bn(res)

        x += res * (self.gating_param if self.re_zero else 1)
        x = self.relu(x)
        return x


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, re_zero=False):
        super().__init__()
        self.re_zero = re_zero
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_channels)
        self.conv3 = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if self.re_zero:
            self.gating_param = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.gating_param, 0.1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity * (self.gating_param if self.re_zero else 1)
        x = self.relu(x)
        return x

