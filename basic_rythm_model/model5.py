
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Model ===
class SEBlock(nn.Module):
    def __init__(self, c, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // reduction), nn.ReLU(), nn.Linear(c // reduction, c), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.se = SEBlock(out_c)
        self.down = nn.Sequential()
        if in_c != out_c or stride != 1:
            self.down = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm1d(out_c))

    def forward(self, x):
        identity = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) + identity
        return self.relu(out)

class SE_ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(130, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(torch.cat([x, demo], dim=1))


# === LSTM-enhanced Model ===
class SE_ResNet1D_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 128, stride=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64*2 + 2, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1)  # [B, T, C] for LSTM
        _, (h_n, _) = self.lstm(x)
        h_out = torch.cat([h_n[0], h_n[1]], dim=1)  # [B, 128]
        return self.fc(torch.cat([h_out, demo], dim=1))

