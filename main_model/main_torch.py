import os
import random
import numpy as np
from scipy.signal import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from main_model.consts import INV_ANNOTATION_MAP, ANNOTATION_MAP, WINDOW_SIZE, EPOCHS, BATCH_SIZE, NUM_CLASSES

# Utility: confusion matrix plot
def plot_conf_matrix(y_true, y_pred, fold):
    cm = confusion_matrix(y_true, y_pred)
    labels = [INV_ANNOTATION_MAP[i] for i in range(len(INV_ANNOTATION_MAP))]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'conf_matrix_fold_{fold}.png')
    plt.close()

# Residual Block for 1D CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ELU()
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# Updated Model using Residual Blocks
class ECGClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = ResidualBlock(1, 64, downsample=True)
        self.block2 = ResidualBlock(64, 64)
        self.block3 = ResidualBlock(64, 128, downsample=True)
        self.block4 = ResidualBlock(128, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).squeeze(-1)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)

# Training function
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    X = np.load('X.npy')
    y = np.load('y.npy')


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = ECGClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    writer = SummaryWriter('runs/ecg_experiment')
    os.makedirs('logs', exist_ok=True)
    log_csv = open('logs/training_log.csv', 'w', newline='')
    csv_writer = csv.writer(log_csv)
    csv_writer.writerow(['epoch', 'train_loss', 'val_acc', 'val_f1', 'val_prec', 'val_rec'])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_batch.numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        val_prec = precision_score(all_targets, all_preds, average='macro')
        val_rec = recall_score(all_targets, all_preds, average='macro')

        print(f"[{epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Precision/val', val_prec, epoch)
        writer.add_scalar('Recall/val', val_rec, epoch)
        csv_writer.writerow([epoch, train_loss, val_acc, val_f1, val_prec, val_rec])
        log_csv.flush()
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'model_best.pt')
            best_val_acc = val_acc

    writer.close()
    log_csv.close()

    model.load_state_dict(torch.load('model_best.pt'))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    print("\nFinal Classification Report:")
    print(classification_report(all_targets, all_preds, labels=list(INV_ANNOTATION_MAP.keys()),
                                target_names=list(INV_ANNOTATION_MAP.values()), zero_division=0))

    plot_conf_matrix(all_targets, all_preds, fold=1)

    dummy = torch.randn(1, 1, WINDOW_SIZE, device=device)
    torch.onnx.export(model, dummy, 'model.onnx',
                      input_names=['input'], output_names=['output'],
                      opset_version=13)
    print('Saved best model and ONNX.')

if __name__ == '__main__':
    train_model()
