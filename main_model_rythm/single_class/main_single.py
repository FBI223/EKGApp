import math
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
import torch.nn.functional as F

DROPOUT_RATE_BLOCK = 0.4
DROPOUT_RATE_FC = 0.7

# === Ustawienia ===
BATCH_SIZE = 64
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NSR_MAX_COUNT= 6000

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
SUBSETS = ['cpsc_2018', 'cpsc_2018_extra' , 'georgia' ,'ptb-xl'   ]

SAVE_NAME = "model_mobile_single"
wavelet = 'bior2.6'
start_thresh_level = 4
threshold_scale = 0.3
threshold_mode = 'soft'

CPSC_CODES = [
    "NSR",          # Prawidłowy rytm zatokowy
    "AF_FLUTTER",   # Migotanie i trzepotanie przedsionków
    "PAC",          # Skurcze przedwczesne przedsionkowe
    "PVC",          # Skurcze przedwczesne komorowe
    "BBB",          # Bloki odnóg (RBBB, LBBB, CRBBB)
    "SVT",          # Nadkomorowe tachyarytmie (AVRT, AT, AVNRT)
    "AV_BLOCK",     # Bloki przedsionkowo-komorowe
    "TORSADES",      # TdP / wydłużony QT

]

CODE_EQUIV = {
    # === NSR ===
    "59118001": "NSR",
    "426783006": "NSR",
    "39732003": "NSR",
    "164896001": "NSR",

    # === AF & FLUTTER ===
    "164889003": "AF_FLUTTER",
    "164890007": "AF_FLUTTER",
    "426995002": "AF_FLUTTER",

    # === PAC ===
    "429622005": "PAC",
    "47665007":  "PAC",
    "17338001":  "PAC",

    # === PVC ===
    "270492004": "PVC",

    # === BBB (łącznie LBBB, RBBB, CRBBB) ===
    "164884008": "BBB",  # LBBB
    "284470004": "BBB",  # RBBB

    # === SVT ===
    "164909002": "SVT",


    # === AV Blocks ===
    "164865005": "AV_BLOCK",  # II degree AVB
    "427393009": "AV_BLOCK",  # III degree AVB
    "59931005": "AV_BLOCK",  # AV dissociation (czasem stosowane)



    # === Torsades de Pointes / Prolonged QT ===
    "10370003": "TORSADES"
}


T_ABNORM_CODES = [
    "164867002", "111975006", "164934002", "445118002", "164930006",
    "164951009", "426177001", "164861001"
]

OTHER_CODES = [
    "39732003", "164934002", "164951009", "445118002", "111975006",
    "164930006", "426177001", "713426002", "59931005", "425623009",
    "164917005", "698252002", "55930002", "713427006", "425419005",
    "251146004", "54329005", "17338001", "164947007", "426434006",
    "164890007", "426627000", "89792004", "63593006", "251120003",
    "445211001", "427172004", "413844008", "251200008", "428417006",
    "446358003", "6374002", "266249003", "11157007", "74390002"
]



N_CLASSES = len(CPSC_CODES)

CLASS2IDX = {code: i for i, code in enumerate(CPSC_CODES)}
CLASS_NAMES = list(CLASS2IDX.keys())  # nazwy = kody


def add_noise(x, noise_level=0.04):
    return x + np.random.normal(0, noise_level, x.shape)

def scale_amplitude(x, scale_range=(0.9, 1.1)):
    return x * np.random.uniform(*scale_range)

def time_shift(x, max_shift=50):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift, axis=-1)

def random_crop(x, target_len=SEG_LEN):
    if x.shape[-1] <= target_len:
        return x
    start = np.random.randint(0, x.shape[-1] - target_len)
    return x[..., start:start + target_len]

def stretch_signal(x, rate_range=(0.95, 1.05)):
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



def augment_signal(x, class_code):
    rare = {"PAC", "BBB", "TORSADES"}
    medium = {"PVC", "AV_BLOCK"}
    common = {"NSR", "AF_FLUTTER", "SVT"}

    if class_code in rare:
        prob = 0.8
    elif class_code in medium:
        prob = 0.5
    elif class_code in common:
        prob = 0.2
    else:
        return x

    if random.random() < prob:
        if class_code in rare:
            x = add_noise(x, 0.1)
            x = scale_amplitude(x, (0.7, 1.3))
            x = time_shift(x, 200)
            x = stretch_signal(x, (0.85, 1.15))
        elif class_code in medium:
            x = add_noise(x, 0.05)
            x = scale_amplitude(x, (0.9, 1.1))
            x = time_shift(x, 100)
        else:
            x = add_noise(x, 0.02)

    return x



def augment_dataset(X, y, d, class_names, multiply_map):
    """
    X         : torch.Tensor (N, 1, L)
    y         : torch.Tensor (N,)
    d         : torch.Tensor (N, 2)
    class_names : list[str] z kodami klas
    multiply_map : dict[int → int] ile augmentacji na 1 próbkę danej klasy

    return: X_aug, y_aug, d_aug – powiększony zbiór
    """
    X_aug = [X]
    y_aug = [y]
    d_aug = [d]

    for class_idx, mult in multiply_map.items():
        class_mask = (y == class_idx)
        X_cls = X[class_mask]
        d_cls = d[class_mask]
        class_code = class_names[class_idx]

        for _ in range(mult):
            for i in range(len(X_cls)):
                x_i = X_cls[i].numpy()
                demo_i = d_cls[i].numpy()

                x_aug = augment_signal(x_i, class_code)
                x_aug = torch.tensor(x_aug, dtype=torch.float32)

                age_aug, sex_aug = perturb_demographics(*demo_i)
                demo_aug = torch.tensor([age_aug, sex_aug], dtype=torch.float32)

                X_aug.append(x_aug.unsqueeze(0))  # (1, L) → (1, 1, L)
                y_aug.append(torch.tensor(class_idx).unsqueeze(0))  # (1,)
                d_aug.append(demo_aug.unsqueeze(0))  # (1, 2)

    # sklej wszystko
    X_final = torch.cat(X_aug)
    y_final = torch.cat(y_aug)
    d_final = torch.cat(d_aug)

    return X_final, y_final, d_final



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
        y = self.pool(x).squeeze(-1)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).unsqueeze(-1)
        return x * y

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

class SE_MobileNet1D_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            CustomSiLU()
        )

        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            SEBlock(32),
            nn.Dropout(DROPOUT_RATE_BLOCK)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            SEBlock(64),
            nn.Dropout(DROPOUT_RATE_BLOCK)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            SEBlock(128),
            nn.Dropout(DROPOUT_RATE_BLOCK)
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(128, 128, stride=1),
            SEBlock(128),
            nn.Dropout(DROPOUT_RATE_BLOCK)
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        # Dodaj pod spodem:
        self.lstm_norm = nn.LayerNorm(128)
        self.lstm_dropout = nn.Dropout(DROPOUT_RATE_BLOCK)

        self.fc = nn.Sequential(
            nn.Linear(2 * 64 + 2, 64),
            nn.BatchNorm1d(64),
            CustomSiLU(),
            nn.Dropout(DROPOUT_RATE_FC),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):
        demo = demo.float()
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (B, 128, L)

        x = x.transpose(1, 2)  # (B, L, 128) for LSTM
        lstm_out, _ = self.lstm(x)  # (B, L, 128)
        lstm_feat = lstm_out[:, -1, :]  # last timestep
        lstm_feat = self.lstm_dropout(self.lstm_norm(lstm_feat))

        # Normalize demographic data
        age = demo[:, 0:1] / 110.0
        sex = demo[:, 1:2]
        demo_norm = torch.cat([age, sex], dim=1)

        x = torch.cat([lstm_feat, demo_norm], dim=1)
        return self.fc(x)




import torch
import torch.nn as nn

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
        y = self.pool(x).squeeze(-1)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).unsqueeze(-1)
        return x * y

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

class SE_MobileNet1D_noLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            CustomSiLU()
        )

        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            SEBlock(32),
            nn.Dropout(0.1)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            SEBlock(64),
            nn.Dropout(0.1)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            SEBlock(128),
            nn.Dropout(0.1)
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(128, 128, stride=1),
            SEBlock(128),
            nn.Dropout(0.1)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.BatchNorm1d(64),
            CustomSiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x).squeeze(-1)  # (B, 128)

        age = demo[:, 0:1] / 110.0
        sex = demo[:, 1:2]
        demo_norm = torch.cat([age, sex], dim=1)

        x = torch.cat([x, demo_norm], dim=1)
        return self.fc(x)




class EarlyStopping:
    def __init__(self, patience=7):
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



def load_and_process_signal(path, seg_len, target_fs ):
    import wfdb
    import pywt
    from scipy.signal import resample

    try:
        sig, meta = wfdb.rdsamp(path)
        fs = meta['fs'] if isinstance(meta, dict) else meta.fs

        if fs != target_fs:
            sig = resample(sig, int(sig.shape[0] * target_fs / fs), axis=0)
        sig = sig.T.astype(np.float32)

        sig = wavelet_denoise(sig)

        if sig.shape[1] < seg_len:
            return None

        sig = sig[:, :seg_len]
        sig = normalize(sig)
        sig = sig[LEAD_IDX:LEAD_IDX+1]  # lead II only
        return sig

    except Exception as e:
        print(f"[ERROR] {path} | {e}")
        return None


def wavelet_denoise(sig):

    wavelet = 'bior2.6'
    start_level = 4
    scale = 0.3
    mode = 'soft'

    denoised = []
    wavelet_len = pywt.Wavelet(wavelet).dec_len

    for lead in sig:
        level = min(8, pywt.dwt_max_level(len(lead), wavelet_len))
        coeffs = pywt.wavedec(lead, wavelet, level=level)

        for i in range(start_level, len(coeffs)):
            if coeffs[i].size == 0 or not np.all(np.isfinite(coeffs[i])):
                continue
            sigma = np.median(np.abs(coeffs[i])) / 0.6745
            if not np.isfinite(sigma) or sigma <= 1e-6:
                continue
            thresh = scale * sigma * np.sqrt(2 * np.log(len(coeffs[i])))
            coeffs[i] = pywt.threshold(coeffs[i], thresh, mode=mode)

        rec = pywt.waverec(coeffs, wavelet)
        denoised.append(rec[:lead.shape[0]])

    return np.vstack(denoised).astype(np.float32)



def load_demo(path):
    age, sex, _ = parse_header(path + '.hea')
    if not np.isfinite(age): age = AGE_MEAN
    if not np.isfinite(sex): sex = 0.0
    return [float(age), float(sex)]



def normalize(sig):
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-8
    return (sig - mean) / std


def load_dataset_in_memory(paths, labels, seg_len=SEG_LEN, target_fs=TARGET_FS,
                           wavelet='bior2.6', start_level=4,
                           scale=0.3, mode='soft'):

    X_list, Y_list, demo_list = [], [], []

    for i, path in enumerate(tqdm(paths)):
        x = load_and_process_signal(path, seg_len, target_fs)
        if x is None or not np.all(np.isfinite(x)):
            continue

        demo = load_demo(path)
        X_list.append(x)
        demo_list.append(demo)
        Y_list.append(labels[i])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    demo = torch.tensor(np.array(demo_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.long)

    return X, demo, Y

class ECGInMemoryDataset(Dataset):
    def __init__(self, X, demo, Y):
        self.X = X
        self.demo = demo
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.demo[idx], self.Y[idx]



def reduce_dominant_class(X, d, y, dominant_class_idx, max_count= NSR_MAX_COUNT ):
    idx_keep = []
    counter = 0
    for i in range(len(y)):
        if y[i].item() == dominant_class_idx:
            if counter >= max_count:
                continue
            counter += 1
        idx_keep.append(i)
    return X[idx_keep], d[idx_keep], y[idx_keep]


def save_dataset_to_npz(X, demo, Y, npz_dir="."):
    np.savez_compressed(os.path.join(npz_dir, "X.npz"), X.numpy())
    np.savez_compressed(os.path.join(npz_dir, "d.npz"), demo.numpy())
    np.savez_compressed(os.path.join(npz_dir, "y.npz"), Y.numpy())


def save_augmented_dataset_to_npz(X, demo, Y, npz_dir="."):
    np.savez_compressed(os.path.join(npz_dir, "Xau.npz"), X.numpy())
    np.savez_compressed(os.path.join(npz_dir, "dau.npz"), demo.numpy())
    np.savez_compressed(os.path.join(npz_dir, "yau.npz"), Y.numpy())


def load_dataset_from_npz(npz_dir="."):
    X = np.load(os.path.join(npz_dir, "X.npz"))["arr_0"]
    d = np.load(os.path.join(npz_dir, "d.npz"))["arr_0"]
    y = np.load(os.path.join(npz_dir, "y.npz"))["arr_0"]
    return torch.tensor(X, dtype=torch.float32), \
           torch.tensor(d, dtype=torch.float32), \
           torch.tensor(y, dtype=torch.long)


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
                    if 'female' in sex_str:
                        sex = 1.0
                    elif 'male' in sex_str:
                        sex = 0.0
                    else:
                        sex = 0.0  # fallback

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

    return float(age), float(sex), codes




def train_single_label_model(model, train_loader, val_loader, num_epochs,
                             criterion, optimizer, scheduler, device,
                             class_names, model_path="model_single.pt"):
    early_stopping = EarlyStopping(patience=7)
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
                probs = torch.softmax(output, dim=1)  # ⬅️ potrzebne do confidence
                conf, pred = torch.max(probs, dim=1)


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
            SAVE_NAME2 = "model_mobile_single.pt"
            torch.save(model.state_dict(), SAVE_NAME2)
            print(f"💾 Zapisano model do {SAVE_NAME2}")


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
            print(" Early stopping triggered")
            break


def build_multiply_map_auto_balanced(y_tensor, class_names, nsr_code="NSR", factor=0.5):
    """
    Dla każdej klasy (poza NSR) ustala ile augmentacji trzeba wykonać,
    by zbliżyć się do docelowej liczności = factor * NSR_count.
    """
    counts = Counter(y_tensor.numpy().tolist())
    nsr_idx = class_names.index(nsr_code)
    nsr_count = counts.get(nsr_idx, 0)

    if nsr_count == 0:
        raise ValueError("Brak klasy NSR w danych!")

    target_size = int(nsr_count * factor)
    multiply_map = {}

    for idx, code in enumerate(class_names):
        if idx == nsr_idx:
            continue  # pomijamy NSR

        current = counts.get(idx, 0)
        if current == 0:
            continue

        needed = target_size - current
        if needed <= 0:
            continue

        mult = math.ceil(needed / current)
        multiply_map[idx] = mult

    return multiply_map




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

def plot_augmented_class_histogram(y_original, y_augmented, class_names, fname="hist_aug.png"):
    from collections import Counter
    import matplotlib.pyplot as plt

    count_original = Counter(y_original.numpy().tolist())
    count_augmented = Counter(y_augmented.numpy().tolist())

    all_classes = list(range(len(class_names)))
    orig_counts = [count_original.get(c, 0) for c in all_classes]
    aug_counts = [count_augmented.get(c, 0) - count_original.get(c, 0) for c in all_classes]

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, orig_counts, label="Oryginalne", color='tab:blue')
    plt.bar(class_names, aug_counts, bottom=orig_counts, label="Augmentowane", color='tab:orange')

    plt.title("Histogram klas po augmentacji")
    plt.ylabel("Liczba próbek")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()



def prepare_dataset(npz_dir=".", nsr_idx=CLASS2IDX["NSR"], nsr_max=NSR_MAX_COUNT,
                    class_names=CLASS_NAMES, val_ratio=VAL_RATIO):
    """
    Sprawdza istnienie danych .npz, wczytuje lub tworzy dane, augmentuje dane treningowe
    i zwraca dane podzielone na train/val + histogramy.
    """

    X_npz = os.path.join(npz_dir, "X.npz")
    y_npz = os.path.join(npz_dir, "y.npz")
    d_npz = os.path.join(npz_dir, "d.npz")

    if os.path.exists(X_npz) and os.path.exists(y_npz) and os.path.exists(d_npz):
        print("✅ Znaleziono .npz – wczytywanie z pamięci")
        X, d, y = load_dataset_from_npz(npz_dir)
    else:
        print("📦 Brak .npz – przetwarzanie sygnałów WFDB...")
        paths, labels = load_single_label_paths_and_labels(CLASS2IDX)

        X, d, y = load_dataset_in_memory(paths, labels)
        X, d, y = reduce_dominant_class(X, d, y, dominant_class_idx=nsr_idx, max_count=nsr_max)
        save_dataset_to_npz(X, d, y, npz_dir)

    # === Podział
    X_train, X_val, d_train, d_val, y_train, y_val = train_test_split(
        X, d, y, test_size=val_ratio, stratify=y, random_state=SEED
    )

    # === Augmentacja
    multiply_map = build_multiply_map_auto_balanced(y_train, class_names=CLASS_NAMES, nsr_code="NSR", factor=0.5)

    X_train_aug, y_train_aug, d_train_aug = augment_dataset(X_train, y_train, d_train, class_names, multiply_map)
    save_augmented_dataset_to_npz(X_train_aug, d_train_aug, y_train_aug, npz_dir)

    # === Histogramy
    plot_class_histogram(y.numpy().tolist(), class_names, fname="hist_all.png")
    plot_augmented_class_histogram(y_train, y_train_aug, class_names, fname="hist_aug_final.png")

    print(f"✅ Dane gotowe: Train: {len(X_train_aug)} | Val: {len(X_val)}")

    return X_train_aug, y_train_aug, d_train_aug, X_val, y_val, d_val

    #return X_train, y_train, d_train, X_val, y_val, d_val



if __name__ == "__main__":
    X_train, y_train, d_train, X_val, y_val, d_val = prepare_dataset()

    train_ds = ECGInMemoryDataset(X_train, d_train, y_train)
    val_ds = ECGInMemoryDataset(X_val, d_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    #model = SE_MobileNet1D_LSTM(num_classes=len(CLASS2IDX)).to(DEVICE)
    model = SE_MobileNet1D_noLSTM(num_classes=len(CLASS2IDX)).to(DEVICE)


    weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

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
        model_path=SAVE_NAME + ".pt"
    )


