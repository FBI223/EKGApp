import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s.expand_as(x)


class BasicBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.se = SEBlock(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SE_ResNet_LSTM(nn.Module):
    def __init__(self, num_classes=12, in_channels=1):
        super().__init__()
        self.in_c = 64

        self.layer0 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_c, blocks, stride):
        downsample = None
        if stride != 1 or self.in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )
        layers = [BasicBlock1D(self.in_c, out_c, stride, downsample)]
        self.in_c = out_c
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):  # x: [B, 1, L]
        x = self.layer0(x)      # → [B, 64, L//4]
        x = self.layer1(x)      # → [B, 64, L//4]
        x = self.layer2(x)      # → [B, 128, L//8]
        x = self.layer3(x)      # → [B, 256, L//16]

        x = x.permute(0, 2, 1)  # → [B, T, C]
        x, _ = self.lstm(x)     # → [B, T, 2*H]
        x = x.mean(dim=1)       # → [B, 256]
        out = self.fc(x)
        return out