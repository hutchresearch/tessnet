import torch
from torch import nn


class LSTM2d(nn.Module):
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

        self.cnn = nn.Sequential(
            self.layer1,
            self.layer2
        )

        self.fc1 = nn.Linear(128, 256)

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)

        if self.period_head:
            self.period_fc = nn.Linear(256, 1)
        if self.classifier_head:
            self.classifier_fc = nn.Linear(256, num_classes)

    def forward(self, x):

        batch_size = x.shape[0]
        num_channels = x.shape[1]
        num_frames = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]

        x = torch.permute(x, (0, 2, 1, 3, 4)).contiguous()

        x = x.view(-1, num_channels, height, width)

        x = self.cnn(x)

        x = x.view(batch_size, num_frames, -1)

        x = self.fc1(x)

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
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.downsample_bn = nn.BatchNorm2d(out_channels)
        if self.re_zero:
            self.gating_param = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.gating_param, 0.1)
            nn.init.zeros_(self.downsample.weight)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

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
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

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

