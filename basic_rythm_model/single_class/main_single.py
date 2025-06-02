
import pywt
import wfdb
from scipy import signal
from scipy.signal import resample
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader
from sklearn.metrics import (
     confusion_matrix, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import random
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler



# === Ustawienia ===
BATCH_SIZE = 64
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EPOCHS = 1
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
SUBSETS = ['cpsc_2018', 'cpsc_2018_extra' , 'georgia'   ]

SAVE_NAME = "model_mobile_single"
wavelet = 'bior2.6'
start_thresh_level = 4
threshold_scale = 0.3
threshold_mode = 'soft'



CPSC_CODES = [
    "59118001",   # NSR – Normal sinus rhythm
    "164889003",  # AF – Atrial fibrillation
    "426783006",  # Bradycardia (including sinus bradycardia)
    "429622005",  # PAC – Premature atrial contraction
    "270492004",  # PVC – Premature ventricular contraction
    "164884008",  # LBBB – Left bundle branch block
    "284470004",  # RBBB – Right bundle branch block
    "164909002",  # SVT – Supraventricular tachycardia
    "428750005",  # IAVB – 1st degree AV block
    "164867002",  # T-wave abnormality
]

N_CLASSES = len(CPSC_CODES)



CODE_EQUIV = {
    "39732003": "59118001",      # Sinus rhythm → NSR
    "164896001": "59118001",     # Alternate sinus rhythm → NSR
    "164931005": "284470004",    # CRBBB → RBBB
    "164873001": "164884008",    # Incomplete LBBB → LBBB
    "17338001": "429622005",     # SVPB → PAC
    "47665007": "429622005",     # Atrial premature beat → PAC
    "164890007": "164889003",    # Atrial flutter → AF
}



CLASS2IDX = {code: i for i, code in enumerate(CPSC_CODES)}
CLASS_NAMES = list(CLASS2IDX.keys())  # nazwy = kody




def add_noise(x, noise_level=0.05):
    return x + np.random.normal(0, noise_level, x.shape)

def scale_amplitude(x, scale_range=(0.8, 1.2)):
    return x * np.random.uniform(*scale_range)

def time_shift(x, max_shift=100):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift, axis=-1)

def random_crop(x, target_len=SEG_LEN):
    if x.shape[-1] <= target_len:
        return x
    start = np.random.randint(0, x.shape[-1] - target_len)
    return x[..., start:start + target_len]

def stretch_signal(x, rate_range=(0.9, 1.1)):
    rate = np.random.uniform(*rate_range)
    stretched = signal.resample(x, int(x.shape[-1] * rate), axis=-1)
    if stretched.shape[-1] > x.shape[-1]:
        return stretched[..., :x.shape[-1]]
    else:
        pad = x.shape[-1] - stretched.shape[-1]
        return np.pad(stretched, ((0, 0), (0, pad)), mode='constant')



def perturb_demographics(age, sex):
    noise = np.random.normal(0, 2.0)
    perturbed_age = np.clip(age + noise, 0, 110)
    return perturbed_age, sex




def augment_signal(x, class_id):
    # prawdopodobieństwo augmentacji zależne od klasy
    if class_id in [2, 6, 9]:  # rzadkie
        prob = 0.8
    elif class_id in [3, 7, 8]:  # średnie
        prob = 0.5
    else:  # częste
        prob = 0.2

    if random.random() < prob:
        if class_id in [2, 6, 9]:
            x = add_noise(x, 0.1)
            x = scale_amplitude(x, (0.7, 1.3))
            x = time_shift(x, 200)
            x = stretch_signal(x, (0.85, 1.15))
        elif class_id in [3, 7, 8]:
            x = add_noise(x, 0.05)
            x = scale_amplitude(x, (0.9, 1.1))
            x = time_shift(x, 100)
        else:
            x = add_noise(x, 0.02)

    return x



class CustomSiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.act = CustomSiLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1)              # (B, C, 1) → (B, C)
        y = self.fc1(y)                           # (B, C // r)
        y = self.act(y)
        y = self.fc2(y)                           # (B, C)
        y = self.sigmoid(y).unsqueeze(-1)         # (B, C) → (B, C, 1)
        return x * y                              # (B, C, L)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = CustomSiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class SE_MobileNet1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            CustomSiLU()
        )

        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            SEBlock(32),
            nn.Dropout(p=0.1)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            SEBlock(64),
            nn.Dropout(p=0.1)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            SEBlock(128),
            nn.Dropout(p=0.1)
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(128, 128, stride=1),
            SEBlock(128),
            nn.Dropout(p=0.1)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(130, 64),
            nn.BatchNorm1d(64),
            CustomSiLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):  # x: (B, 1, L), demo: (B, 2)
        x = self.stem(x)         # (B, 16, L/2)
        x = self.block1(x)       # (B, 32, L/4)
        x = self.block2(x)       # (B, 64, L/8)
        x = self.block3(x)       # (B, 128, L/16)
        x = self.block4(x)       # (B, 128, L/16)
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = torch.cat([x, demo], dim=1)  # (B, 130)
        return self.fc(x)


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


def plot_aug_histogram(counter, class_names, fname="hist_aug.png"):
    counts = [counter.get(i, 0) for i in range(len(class_names))]
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, counts)
    plt.xticks(rotation=45)
    plt.title("Histogram augmentacji klas")
    plt.ylabel("Liczba augmentowanych próbek")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


# === Dataset z demografią ===
class ECGDatasetSingleLabelWithDemo(Dataset):
    def __init__(self, paths, labels, seg_len=SEG_LEN, target_fs=TARGET_FS,
                 training=False, wavelet='bior2.6', start_thresh_level=4,
                 threshold_scale=0.3, threshold_mode='soft'):
        self.paths = paths
        self.labels = labels
        self.seg_len = seg_len
        self.target_fs = target_fs
        self.training = training
        self.augmented_counter = Counter()

        # Denoising parameters
        self.wavelet = wavelet
        self.start_thresh_level = start_thresh_level
        self.threshold_scale = threshold_scale
        self.threshold_mode = threshold_mode

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

    def wavelet_denoise(self, sig):
        denoised = []
        wavelet_len = pywt.Wavelet(self.wavelet).dec_len

        for lead in sig:
            max_level = pywt.dwt_max_level(len(lead), wavelet_len)
            level = min(8, max_level)
            coeffs = pywt.wavedec(lead, wavelet=self.wavelet, level=level)

            for i in range(self.start_thresh_level, len(coeffs)):
                if coeffs[i].size == 0 or not np.all(np.isfinite(coeffs[i])):
                    continue
                sigma = np.median(np.abs(coeffs[i])) / 0.6745
                if not np.isfinite(sigma) or sigma <= 1e-6:
                    continue
                threshold = self.threshold_scale * sigma * np.sqrt(2 * np.log(len(coeffs[i])))
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode=self.threshold_mode)

            denoised_lead = pywt.waverec(coeffs, wavelet=self.wavelet)
            denoised.append(denoised_lead[:lead.shape[0]])

        return np.vstack(denoised).astype(np.float32)

    def trunc(self, sig):
        if sig.shape[1] < self.seg_len:
            return None
        return sig[:, :self.seg_len]

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            sig, fs = self.load_record(path)
            sig = self.resample_sig(sig, fs)
            sig = self.wavelet_denoise(sig)
            sig = self.trunc(sig)
            if sig is None:
                return None
            sig = self.normalize(sig)

            if not np.all(np.isfinite(sig)):
                return None

            age, sex, _ = parse_header(path + '.hea')
            if not np.isfinite(age): age = AGE_MEAN
            if not np.isfinite(sex): sex = 0.0

            if self.training:
                original_sig = sig.copy()
                sig_aug = augment_signal(sig, label)
                if not np.array_equal(sig_aug, original_sig):
                    self.augmented_counter[label] += 1
                sig = sig_aug
                age, sex = perturb_demographics(age, sex)

            x_tensor = torch.tensor(sig[LEAD_IDX:LEAD_IDX + 1], dtype=torch.float32)
            demo_tensor = torch.tensor([age, sex], dtype=torch.float32)
            y_tensor = torch.tensor(label, dtype=torch.long)

            return x_tensor, demo_tensor, y_tensor
        except Exception as e:
            print(f"[ERROR] failed index {idx} | path: {path} | err: {e}")
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
                             class_names, model_path="model_single.pt"):
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

    print("Występujące klasy:", sorted(np.unique(labels)))
    print("CLASS2IDX:", CLASS2IDX)

    # === Dataset & Loader ===
    train_ds = ECGDatasetSingleLabelWithDemo(train_paths, train_labels, training=True)
    val_ds = ECGDatasetSingleLabelWithDemo(val_paths, val_labels, training=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_none)

    # Zlicz klasy
    label_counts = Counter(train_labels)
    total_count = sum(label_counts.values())

    # Waga odwrotna do liczności
    weights_sampler = {cls: total_count / count for cls, count in label_counts.items()}
    sample_weights_sampler = [weights_sampler[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights_sampler,
        num_samples=len(train_labels),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_skip_none
    )

    # === Model ===
    #model = SE_ResNet1D(num_classes=len(CLASS2IDX)).to(DEVICE)
    #model = SE_ResNet1D_LSTM(num_classes=len(CLASS2IDX)).to(DEVICE)
    model = SE_MobileNet1D(num_classes=len(CLASS2IDX)).to(DEVICE)

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
        model_path= SAVE_NAME + ".pt"
    )


    # === Histogram augmentacji po treningu ===
    print("\n📊 Histogram augmentacji klas:")
    for cls_idx in sorted(set(train_labels)):
        count = train_ds.augmented_counter[cls_idx]
        print(f"  {CLASS_NAMES[cls_idx]} ({cls_idx}): {count}")

    plot_aug_histogram(train_ds.augmented_counter, CLASS_NAMES)
