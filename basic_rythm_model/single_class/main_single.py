
import pywt
import wfdb
from scipy.signal import resample
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt

# === Ustawienia ===
BATCH_SIZE = 64
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LR=1e-3
TARGET_FS = 500
LEAD_IDX = 1
SEG_DUR = 10
SEG_LEN = TARGET_FS * SEG_DUR  # 5000 samples
AGE_MEAN = 60.0

BASE_PATH = "/home/msztu223/PycharmProjects/ECG_PROJ/databases/challenge2020_data/training"
SUBSETS = ['cpsc_2018', 'cpsc_2018_extra'   ]


wavelet = 'bior2.6'
start_thresh_level = 4
threshold_scale = 0.3
threshold_mode = 'soft'




CPSC_CODES = [
    "59118001",   # NSR – Normal sinus rhythm
    "164889003",  # LBBB – Left bundle branch block
    "426783006",  # IAVB – 1st degree AV blockdis
    "429622005",  # SVT – Supraventricular tachycardia
    "270492004",  # PVC – Premature ventricular contraction
    "164884008",  # TINV – T wave inversion
    "164909002",  # AF – Atrial fibrillation ✅
    "428750005",  # PAC – Premature atrial contractions
]



CODE_EQUIV = {
    "39732003": "59118001",     # Sinus rhythm → NSR
    "164896001": "59118001",    # Alternate NSR code
    "17338001": "428750005",    # SVPB → PAC
    "47665007": "428750005",    # Atrial premature beat → PAC
    "284470004": "164884008",   # Alt code for TINV
}


CLASS2IDX = {code: i for i, code in enumerate(CPSC_CODES)}
CLASS_NAMES = list(CLASS2IDX.keys())  # nazwy = kody




class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True



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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return self.relu(out)


class SE_ResNet1D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):  # x: (B, 1, L), demo: (B, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)  # (B, 128, 1)
        x = torch.reshape(x, (x.size(0), -1))  # (B, 128)
        x = torch.cat([x, demo], dim=1)  # (B, 130)
        return self.fc(x)


# === Dataset z demografią ===
class ECGDatasetSingleLabelWithDemo(Dataset):
    def __init__(self, paths, labels, seg_len=SEG_LEN, target_fs=TARGET_FS):
        self.paths = paths
        self.labels = labels
        self.seg_len = seg_len
        self.target_fs = target_fs

    def __len__(self):
        return len(self.paths)

    def load_record(self, path):
        x, meta = wfdb.rdsamp(path)
        fs = meta['fs'] if isinstance(meta, dict) else meta.fs
        return x.T.astype(np.float32), fs

    def resample_sig(self, sig, fs):
        if fs != self.target_fs:
            n = int(sig.shape[1] * self.target_fs / fs)
            sig = resample(sig, n, axis=1)
        return sig

    def normalize(self, sig):
        mean = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-8
        return (sig - mean) / std


    def wavelet_denoise(self, sig, wavelet='bior2.6', start_thresh_level=4, threshold_scale=0.5, threshold_mode='hard'):

        denoised = []
        wavelet_len = pywt.Wavelet(wavelet).dec_len

        for lead in sig:
            max_level = pywt.dwt_max_level(len(lead), wavelet_len)
            level = min(8, max_level)
            coeffs = pywt.wavedec(lead, wavelet=wavelet, level=level)

            for i in range(start_thresh_level, len(coeffs)):
                if coeffs[i].size == 0 or not np.all(np.isfinite(coeffs[i])):
                    continue
                sigma = np.median(np.abs(coeffs[i])) / 0.6745
                if not np.isfinite(sigma) or sigma <= 1e-6:
                    continue
                threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(coeffs[i])))
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_mode)

            denoised_lead = pywt.waverec(coeffs, wavelet=wavelet)
            denoised.append(denoised_lead[:lead.shape[0]])

        return np.vstack(denoised)

    def trunc(self, sig):
        if sig.shape[1] < self.seg_len:
            return None  # odrzuć zbyt krótkie
        return sig[:, :self.seg_len]

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            sig, fs = self.load_record(path)
            sig = self.resample_sig(sig, fs)
            sig = self.wavelet_denoise(sig,wavelet=wavelet,start_thresh_level=start_thresh_level,threshold_scale=threshold_scale,threshold_mode=threshold_mode)
            #sig = self.wavelet_denoise_dynamic(sig, wavelet=self.wavelet, start_thresh_level=self.start_thresh_level)

            sig = self.trunc(sig)
            sig = self.normalize(sig)

            if np.isnan(sig).any() or not np.isfinite(sig).all():
                return None

            age, sex, _ = parse_header(path + '.hea')
            if not np.isfinite(age) or not np.isfinite(sex):
                age, sex = AGE_MEAN, 0.0

            x_tensor = torch.tensor(sig[LEAD_IDX:LEAD_IDX+1], dtype=torch.float32)
            demo_tensor = torch.tensor([age, sex], dtype=torch.float32)
            y_tensor = torch.tensor(label, dtype=torch.long)

            return x_tensor, demo_tensor, y_tensor
        except:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def plot_signal_lengths(paths, title="Histogram długości sygnałów", fname="sig_lengths.png"):
    lengths = []
    for path in paths:
        try:
            x, meta = wfdb.rdsamp(path)
            lengths.append(x.shape[0])
        except:
            continue
    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=50)
    plt.title(title)
    plt.xlabel("Liczba próbek")
    plt.ylabel("Liczba rekordów")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def load_single_label_paths_and_labels(class2idx):

    paths = []
    labels = []

    for subset in SUBSETS:
        subset_dir = os.path.join(BASE_PATH, subset)
        for fname in os.listdir(subset_dir):
            if not fname.endswith(".hea"):
                continue

            rec_id = fname[:-4]
            rec_path = os.path.join(subset_dir, rec_id)
            header_path = rec_path + ".hea"

            try:
                _, _, codes = parse_header(header_path)
            except Exception as e:
                continue

            # Zostaw tylko dozwolone klasy
            codes = [c for c in codes if c in class2idx]
            if len(codes) != 1:
                continue  # pomiń multi-label lub brakujące etykiety

            label_idx = class2idx[codes[0]]
            paths.append(rec_path)
            labels.append(label_idx)

    return paths, labels





def parse_header_old(header_path):
    age, sex, codes = None, None, []

    try:
        with open(header_path, 'r') as f:
            for line in f:
                if line.startswith('# Age:'):
                    try:
                        age_val = float(line.split(':')[1].strip())
                        if np.isnan(age_val) or age_val < 0 or age_val > 110:
                            #print(f"[AGE] Invalid age in {header_path}: {age_val}")
                            age = None
                        else:
                            age = age_val
                    except:
                        age = None
                elif line.startswith('# Sex:'):
                    sex_str = line.split(':')[1].strip().lower()
                    sex = 1.0 if 'female' in sex_str else 0.0
                elif 'Dx:' in line:
                    raw = line.split(':')[1].split(',')
                    codes = [CODE_EQUIV.get(c.strip(), c.strip()) for c in raw]
    except Exception as e:
        print(f"[ERROR] parse_header failed for {header_path}: {e}")

    # fallback
    if age is None:
        age = AGE_MEAN
    if sex is None:
        sex = 0.0

    return age, sex, codes


def parse_header(header_path):
    age, sex, codes = None, None, []

    try:
        with open(header_path, 'r') as f:
            for line in f:
                if line.startswith('# Age:'):
                    try:
                        age_val = float(line.split(':')[1].strip())
                        if np.isnan(age_val) or age_val < 0 or age_val > 110:
                            age = None
                        else:
                            age = age_val
                    except:
                        age = None
                elif line.startswith('# Sex:'):
                    sex_str = line.split(':')[1].strip().lower()
                    sex = 1.0 if 'female' in sex_str else 0.0
                elif 'Dx:' in line:
                    raw = line.split(':')[1].split(',')
                    codes = [CODE_EQUIV.get(c.strip(), c.strip()) for c in raw]
    except Exception as e:
        print(f"[ERROR] parse_header failed for {header_path}: {e}")


    # fallback na brak wieku/płci
    if age is None:
        age = AGE_MEAN
    if sex is None:
        sex = 0.0

    return age, sex, codes



def train_single_label_model(model, train_loader, val_loader, num_epochs,
                             criterion, optimizer, scheduler, device,
                             class_names, model_path="model_multi.pt"):
    early_stopping = EarlyStopping(patience=5)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, demo, y in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]"):
            x, demo, y = x.to(device), demo.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, demo)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss, all_y, all_preds = 0.0, [], []
        with torch.no_grad():
            for x, demo, y in tqdm(val_loader, desc=f"Epoch {epoch + 1} [val]"):
                x, demo, y = x.to(device), demo.to(device), y.to(device)
                output = model(x, demo)
                loss = criterion(output, y)
                val_loss += loss.item()
                pred = torch.argmax(output, dim=1)
                all_y.append(y.cpu())
                all_preds.append(pred.cpu())

        avg_val_loss = val_loss / len(val_loader)
        all_y = torch.cat(all_y).numpy()
        all_preds = torch.cat(all_preds).numpy()

        acc = (all_y == all_preds).mean()
        prec = precision_score(all_y, all_preds, average=None, zero_division=0)
        rec = recall_score(all_y, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_y, all_preds, average=None, zero_division=0)
        f1_macro = f1_score(all_y, all_preds, average="macro")
        f1_weighted = f1_score(all_y, all_preds, average="weighted")

        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}")
        print("\nPer-class metrics:")
        for i, name in enumerate(class_names):
            print(f"{name:<6} | Precision: {prec[i]:.3f} | Recall: {rec[i]:.3f} | F1: {f1[i]:.3f}")
        print(f"\nMacro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")

        # === Zapis najlepszego modelu ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print("💾 Zapisano nowy najlepszy model")

        # === Rysuj histogram predykcji ===
        counts = np.bincount(all_preds, minlength=len(class_names))
        plt.figure(figsize=(8, 4))
        plt.bar(class_names, counts)
        plt.xticks(rotation=45)
        plt.ylabel("Predicted count")
        plt.title(f"Prediction histogram – Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(f"hist_epoch_{epoch + 1}.png")
        plt.close()

        # === Confusion Matrix ===
        cm = confusion_matrix(all_y, all_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(include_values=True, xticks_rotation=45, cmap='Blues')
        plt.title(f"Confusion Matrix – Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(f"cm_epoch_{epoch + 1}.png")
        plt.close()

        # === Scheduler i EarlyStopping ===
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        if early_stopping.stop:
            print("⛔️ Early stopping triggered")
            break



def plot_class_histogram(labels, class_names, fname="histogram_klas.png"):
    label_counts = Counter(labels)
    keys_sorted = sorted(set(labels))
    plt.figure(figsize=(8, 4))
    plt.bar([class_names[k] for k in keys_sorted],
            [label_counts[k] for k in keys_sorted])
    plt.xticks(rotation=45)
    plt.title("Histogram liczności klas (po filtracji)")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    # === Załaduj dane ===
    print("📦 Ładowanie danych...")
    paths, labels = load_single_label_paths_and_labels(CLASS2IDX)
    total = len(paths)

    # === Podział train/val ===
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=VAL_RATIO, random_state=SEED, stratify=labels
    )

    plot_class_histogram(labels, CLASS_NAMES, fname="histogram_klas_przed_trenowaniem.png")

    print(f"✅ Train: {len(train_paths)} | Val: {len(val_paths)} | Total: {total}")

    # === Dataset & Loader ===
    train_ds = ECGDatasetSingleLabelWithDemo(train_paths, train_labels)
    val_ds = ECGDatasetSingleLabelWithDemo(val_paths, val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_none)

    # === Model ===
    model = SE_ResNet1D(num_classes=len(CLASS2IDX)).to(DEVICE)
    #model = SE_ResNet1D_LSTM(num_classes=len(CLASS2IDX)).to(DEVICE)

    # === Loss i optymalizator ===
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # === Trening ===
    train_single_label_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        class_names=CLASS_NAMES,
        model_path="model_single.pt"
    )

