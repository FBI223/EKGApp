

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import csv
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from main_model.consts import BATCH_SIZE, EPOCHS, NUM_CLASSES, WINDOW_SIZE, INV_ANNOTATION_MAP, FOLDS
import random

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

    # Apply SMOTE exactly as in the paper
    X = X.reshape((X.shape[0], -1))  # Flatten to (samples, features)
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X = X_resampled.reshape((-1, WINDOW_SIZE, 1))
    y = y_resampled

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



















def plot_avg_metric(fold_acc, fold_f1, output_dir):
    plt.figure()
    plt.bar(['Acc', 'F1'], [np.mean(fold_acc), np.mean(fold_f1)], yerr=[np.std(fold_acc), np.std(fold_f1)], capsize=5)
    plt.title("Average Accuracy and F1 Score Across Folds")
    plt.savefig(os.path.join(output_dir, "avg_metrics.png"))
    plt.close()




def plot_conf_matrix(y_true, y_pred, fold, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    labels = [INV_ANNOTATION_MAP[i] for i in range(len(INV_ANNOTATION_MAP))]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'conf_matrix_fold{fold}.png'))
    plt.close()


def plot_metrics(train_losses, val_accuracies, fold, output_dir):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title(f'Training Metrics Fold {fold}')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'metrics_fold{fold}.png'))
    plt.close()


def export_onnx_model(model, output_path):
    dummy = torch.randn(1, 1, WINDOW_SIZE, device=next(model.parameters()).device)
    torch.onnx.export(model, dummy, output_path, input_names=['input'], output_names=['output'], opset_version=13)



def train_kfold_model(X, y, folds=FOLDS, output_dir='training'):

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(output_dir, exist_ok=True)
    dirs = {
        "onnx": os.path.join(output_dir, "onnx"),
        "pt": os.path.join(output_dir, "pt"),
        "conf_matrix": os.path.join(output_dir, "conf_matrix"),
        "metrics": os.path.join(output_dir, "metrics"),
        "csv": os.path.join(output_dir, "csv"),
        "config": os.path.join(output_dir, "config"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(dirs["config"], 'config.txt'), 'w') as f:
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"EPOCHS={EPOCHS}\n")
        f.write(f"FOLDS={folds}\n")
        f.write(f"SMOTE=not majority\n")
        f.write(f"WINDOW_SIZE={WINDOW_SIZE}\n")
        f.write(f"MODEL=ECGClassifier with ResidualBlocks\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    all_val_acc, all_val_f1 = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🧪 Fold {fold}/{folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

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
        train_losses, val_accuracies = [], []
        f1_macro_list, prec_macro_list, recall_macro_list = [], [], []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().item()
            train_loss = loss_sum / len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    targets.extend(y_batch.numpy())

            val_acc = accuracy_score(targets, preds)
            val_f1 = f1_score(targets, preds, average='macro')
            val_prec = precision_score(targets, preds, average='macro')
            val_rec = recall_score(targets, preds, average='macro')

            val_accuracies.append(val_acc)
            f1_macro_list.append(val_f1)
            prec_macro_list.append(val_prec)
            recall_macro_list.append(val_rec)

            scheduler.step(1 - val_acc)

            if val_acc > best_val_acc:
                torch.save(model.state_dict(), os.path.join(dirs["pt"], f'model_fold{fold}.pt'))
                export_onnx_model(model, os.path.join(dirs["onnx"], f'model_fold{fold}.onnx'))
                best_val_acc = val_acc

            print(f"[Fold {fold}] Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        all_val_acc.append(best_val_acc)
        all_val_f1.append(val_f1)

        plot_conf_matrix(targets, preds, fold, dirs["conf_matrix"])
        plot_metrics(train_losses, val_accuracies, fold, dirs["metrics"])

        with open(os.path.join(dirs["csv"], f'metrics_fold{fold}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_acc', 'val_f1_macro', 'val_prec_macro', 'val_recall_macro'])
            for i in range(EPOCHS):
                writer.writerow([
                    i+1, train_losses[i], val_accuracies[i],
                    f1_macro_list[i], prec_macro_list[i], recall_macro_list[i]
                ])

    plot_avg_metric(all_val_acc, all_val_f1, dirs["metrics"])
    print(f"\n✅ Średnia dokładność: {np.mean(all_val_acc):.4f}, F1: {np.mean(all_val_f1):.4f}")

# W __main__:
if __name__ == '__main__':
    from imblearn.over_sampling import SMOTE
    X = np.load('X.npy')
    y = np.load('y.npy')
    X = X.reshape((X.shape[0], -1))
    X, y = SMOTE(sampling_strategy='not majority', random_state=42).fit_resample(X, y)
    X = X.reshape((-1, WINDOW_SIZE, 1))

    train_kfold_model(X, y, output_dir='training', folds=FOLDS)







