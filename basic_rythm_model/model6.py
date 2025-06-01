import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1)
        return x * y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

class SE_MobileNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            SEBlock(32)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            SEBlock(64)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            SEBlock(128)
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(128, 128, stride=1),
            SEBlock(128)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):  # x: (B, 1, L), demo: (B, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)  # (B, 128, 1)
        x = torch.reshape(x, (x.size(0), -1))  # (B, 128)
        x = torch.cat([x, demo], dim=1)  # (B, 130)
        return self.fc(x)
