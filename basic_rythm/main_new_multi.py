from collections import Counter
import pywt
import wfdb
from scipy.signal import resample
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from model5 import SE_ResNet1D
import random
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import random


# === Ustawienia ===
BATCH_SIZE = 64
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


MAX_SAMPLES=4000
LR=1e-3
TARGET_FS = 500


LEAD_IDX = 1
SEG_DUR = 10
SEG_LEN = TARGET_FS * SEG_DUR  # 5000 samples
AGE_MEAN = 60.0

BASE_PATH = "/home/msztu223/PycharmProjects/ECG_PROJ/databases/challenge2020_data/training"
SUBSETS = ['cpsc_2018', 'cpsc_2018_extra' , 'ptb-xl' , 'georgia']





# === Mapowanie kodów równoważnych (używane tylko przy normalizacji etykiet) ===
CODE_EQUIV = {
    "17338001": "428750005",   # SVPB → PAC
    "47665007": "428750005",   # Atrial premature beat → PAC
    "164890007": "164889003",  # CRBBB → LBBB
    "39732003": "59118001",    # Sinus rhythm → NSR (główny kod do NSR!)
    "164896001": "59118001",   # NSR (alternatywny kod, też NSR)
    "733534002": "284470004",  # MI with STD → STD
    "111975006": "164873001",  # STEMI → STE
}



CPSC_CODES = [
    "426783006",  # IAVB – najczęstsza
    "164865005",  # T-wave abnormality
    "164934002",  # Atrial fibrillation
    "164873001",  # ST elevation
    "428750005",  # Premature atrial contractions
    "164889003",  # Left bundle branch block
    "164861001",  # Low QRS voltage
    "59118001",   # Normal sinus rhythm
    "270492004",  # Premature ventricular contractions
    "427084000",  # Right bundle branch block
    "284470004",  # ST depression
    "429622005",  # Sinus bradycardia
    "164884008",  # T wave inversion
    "164867002",  # Sinus tachycardia
    "164930006",  # 1st degree AV block
]

CLASS2IDX = {code: i for i, code in enumerate(CPSC_CODES)}
CLASS_NAMES = list(CLASS2IDX.keys())  # nazwy = kody






# === Zmienione: PARSER nagłówków uwzględnia CODE_EQUIV i filtruje tylko interesujące ===
def parse_header_filtered(header_path, allowed_codes=None):
    age, sex, codes = None, None, []
    try:
        with open(header_path, 'r') as f:
            for line in f:
                if line.startswith('# Age:'):
                    try:
                        age_val = float(line.split(':')[1].strip())
                        age = age_val if 0 < age_val < 110 else None
                    except: age = None
                elif line.startswith('# Sex:'):
                    sex_str = line.split(':')[1].strip().lower()
                    sex = 1.0 if 'female' in sex_str else 0.0
                elif 'Dx:' in line:
                    raw = line.split(':')[1].split(',')
                    codes = [CODE_EQUIV.get(c.strip(), c.strip()) for c in raw]
                    if allowed_codes:
                        codes = [c for c in codes if c in allowed_codes]
    except: return AGE_MEAN, 0.0, []
    if age is None: age = AGE_MEAN
    if sex is None: sex = 0.0
    return age, sex, codes


# === Zmienione: ładowanie multi-label paths + tylko MULTILABEL_CLASSES ===
def load_multilabel_paths_and_labels_filtered():
    paths, labels = [], []
    for subset in SUBSETS:
        subset_dir = os.path.join(BASE_PATH, subset)
        for fname in os.listdir(subset_dir):
            if not fname.endswith(".hea"): continue
            rec_id = fname[:-4]
            rec_path = os.path.join(subset_dir, rec_id)
            age, sex, codes = parse_header_filtered(rec_path + ".hea", allowed_codes=CLASS2IDX)
            label = np.zeros(len(CLASS2IDX), dtype=np.float32)
            for code in codes:
                if code in CLASS2IDX:
                    label[CLASS2IDX[code]] = 1.0
            if label.sum() == 0: continue
            paths.append(rec_path)
            labels.append(label)
    return paths, labels



def plot_class_distribution(labels, class_codes, title="Histogram liczności klas", filename="class_histogram.png"):

    labels_arr = np.array(labels)
    counts = np.sum(labels_arr, axis=0)

    plt.figure(figsize=(12, 5))
    plt.bar(class_codes, counts)
    plt.title(title)
    plt.ylabel("Liczba wystąpień")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()





def parse_header(header_path):
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



class ECGDatasetMultiLabelWithDemo(Dataset):
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

    def wavelet_denoise(self, sig, wavelet='bior2.6'):
        return sig  # opcjonalnie wyłączone

    def pad_or_trunc(self, sig):
        if sig.shape[1] < self.seg_len:
            return None  # pomiń zbyt krótkie
        return sig[:, :self.seg_len]

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            sig, fs = self.load_record(path)
            sig = self.resample_sig(sig, fs)
            sig = self.wavelet_denoise(sig)
            sig = self.pad_or_trunc(sig)
            if sig is None or np.isnan(sig).any() or not np.isfinite(sig).all():
                return None

            sig = self.normalize(sig)
            age, sex, _ = parse_header(path + '.hea')
            if not np.isfinite(age) or not np.isfinite(sex):
                age, sex = AGE_MEAN, 0.0

            x_tensor = torch.tensor(sig[LEAD_IDX:LEAD_IDX+1], dtype=torch.float32)
            demo_tensor = torch.tensor([age, sex], dtype=torch.float32)
            y_tensor = torch.tensor(label, dtype=torch.float32)

            return x_tensor, demo_tensor, y_tensor
        except:
            return None



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

    def wavelet_denoise(self, sig, wavelet='bior2.6'):

        return sig  # <- Tymczasowo wyłącz denoising

        denoised = []
        wavelet_len = pywt.Wavelet(wavelet).dec_len
        for lead in sig:
            max_level = pywt.dwt_max_level(len(lead), wavelet_len)
            level = min(8, max_level)
            coeffs = pywt.wavedec(lead, wavelet=wavelet, level=level)
            coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
            denoised_lead = pywt.waverec(coeffs, wavelet=wavelet)
            denoised.append(denoised_lead[:lead.shape[0]])
        return np.vstack(denoised)

    def pad_or_trunc(self, sig):
        L = sig.shape[1]
        if L < self.seg_len:
            pad = np.zeros((sig.shape[0], self.seg_len - L), dtype=sig.dtype)
            sig = np.concatenate([sig, pad], axis=1)
        else:
            sig = sig[:, :self.seg_len]
        return sig

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            sig, fs = self.load_record(path)
            sig = self.resample_sig(sig, fs)
            sig = self.wavelet_denoise(sig)
            sig = self.pad_or_trunc(sig)
            if sig.shape[1] < self.seg_len:
                return None
            sig = self.normalize(sig)

            if np.isnan(sig).any() or not np.isfinite(sig).all():
                return None

            age, sex, _ = parse_header(path + '.hea')
            if not np.isfinite(age) or not np.isfinite(sex):
                age, sex = 60.0, 0.0

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


def load_single_label_paths_and_labels(class2idx, allowed_codes=None):
    """
    Wczytuje tylko próbki z dokładnie jedną klasą (z allowed_codes).
    """
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
                print(f"[ERROR] {header_path}: {e}")
                continue

            # Zostaw tylko dozwolone klasy
            codes = [c for c in codes if c in class2idx]
            if len(codes) != 1:
                continue  # pomiń multi-label lub brakujące klasy

            label_idx = class2idx[codes[0]]
            paths.append(rec_path)
            labels.append(label_idx)

    return paths, labels



def load_multilabel_paths_and_labels():
    paths = []
    labels = []

    for subset in SUBSETS:
        subset_dir = os.path.join(BASE_PATH, subset)
        for fname in os.listdir(subset_dir):
            if not fname.endswith(".hea"):
                continue

            rec_id = fname[:-4]  # usuń .hea
            rec_path = os.path.join(subset_dir, rec_id)

            try:
                _, _, codes = parse_header(rec_path + ".hea")
            except Exception as e:
                print(f"[ERROR] {rec_path}: {e}")
                continue

            # konwertuj na multi-hot względem CPSC_CODES
            label = np.zeros(len(CPSC_CODES), dtype=np.float32)
            for code in codes:
                if code in CLASS2IDX:
                    label[CLASS2IDX[code]] = 1.0

            if label.sum() == 0:
                continue  # pomiń jeśli brak interesujących klas

            paths.append(rec_path)
            labels.append(label)

    return paths, labels





# === Funkcja pomocnicza: wybór jednej etykiety z multi-hot na podstawie priorytetu ===
def reduce_to_single_label(multi_hot_labels, class_priority):
    class2idx = {code: i for i, code in enumerate(class_priority)}
    reduced = []
    for label in multi_hot_labels:
        indices = np.where(label)[0]
        if len(indices) == 0:
            reduced.append(class2idx["426177001"])  # fallback: NSR
        else:
            # wybierz pierwszą w kolejności ważności
            reduced.append(indices[0])
    return reduced


def train_single_label_model(model, train_loader, val_loader, num_epochs,
                             criterion, optimizer, scheduler, device,
                             class_names):
    early_stopping = EarlyStopping(patience=5)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, demo, y in train_loader:
            x, demo, y = x.to(device), demo.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, demo)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss, all_y, all_preds = 0.0, [], []
        with torch.no_grad():
            for x, demo, y in val_loader:
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

        # Accuracy
        acc = (all_y == all_preds).mean()

        # Precision, Recall, F1
        prec = precision_score(all_y, all_preds, average=None, zero_division=0)
        rec = recall_score(all_y, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_y, all_preds, average=None, zero_division=0)

        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {acc:.4f}")
        print("\nPer-class metrics:")
        for i, name in enumerate(class_names):
            print(f"{name:<6} | Precision: {prec[i]:.3f} | Recall: {rec[i]:.3f} | F1: {f1[i]:.3f}")

        # Macro + Weighted avg
        f1_macro = f1_score(all_y, all_preds, average="macro")
        f1_weighted = f1_score(all_y, all_preds, average="weighted")
        print(f"\nMacro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")

        # Histogram liczności predykcji
        counts = np.bincount(all_preds, minlength=len(class_names))
        plt.figure(figsize=(8, 4))
        plt.bar(class_names, counts)
        plt.xticks(rotation=45)
        plt.ylabel("Predicted count")
        plt.title(f" Prediction histogram – Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(f"hist_pred_epoch_{epoch + 1}.png")
        plt.show()

        # Confusion matrix
        cm = confusion_matrix(all_y, all_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(include_values=True, xticks_rotation=45, cmap='Blues')
        plt.title(f"Confusion Matrix – Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(f"cm_epoch_{epoch + 1}.png")
        plt.show()

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        if early_stopping.stop:
            print("⛔️ Early stopping triggered")
            break

    torch.save(model.state_dict(), "ecg_resnet1d_best_singlelabel.pt")





'''

if __name__ == "__main__":
    print("📦 Ładowanie ścieżek i etykiet...")
    paths, single_labels = load_single_label_paths_and_labels(CLASS2IDX)

    # 🔻 Redukcja WSZYSTKICH klas do cap=1000
    paths, single_labels = reduce_all_classes_occurrences_singlelabel(
        paths, single_labels, class2idx=CLASS2IDX, cap=1000
    )

    # 📊 Histogram po redukcji do 1000
    plot_class_distribution_single(
        single_labels, CLASS_NAMES,
        title="Po redukcji – max 1000 próbek/klasa",
        fname="hist_reduced_1000.png"
    )

    # ↓ (opcjonalnie) Dodatkowa redukcja dla wybranych klas, jeśli chcesz
    # paths, single_labels = reduce_class_occurrences_singlelabel(paths, single_labels, "426783006", MAX_SAMPLES, CLASS2IDX)

    # === Split stratified ===
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths, single_labels, test_size=TEST_RATIO, random_state=SEED, stratify=single_labels
    )

    # === Dataset i DataLoadery ===
    train_ds = ECGDatasetSingleLabelWithDemo(paths_train, labels_train, seg_len=SEG_LEN)
    val_ds = ECGDatasetSingleLabelWithDemo(paths_val, labels_val, seg_len=SEG_LEN)

    # 📊 Histogram klas w train+val
    plot_class_distribution_single(
        single_labels, CLASS_NAMES,
        title="Single-label only – Distribution",
        fname="hist_singlelabel_cleaned.png"
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_skip_none)

    # === Model i trening ===
    model = SE_ResNet1D(num_classes=len(CLASS_NAMES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4)

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=labels_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_single_label_model(
        model, train_loader, val_loader,
        NUM_EPOCHS, criterion, optimizer, scheduler,
        DEVICE, class_names=CLASS_NAMES
    )

'''

def reduce_class_occurrences(paths, labels, target_code, max_count, class2idx):
    """
    Redukuje liczbę wystąpień danej klasy (multi-label).
    """
    target_idx = class2idx[target_code]
    indices_with_class = [i for i, label in enumerate(labels) if label[target_idx] == 1.0]
    indices_without_class = [i for i, label in enumerate(labels) if label[target_idx] == 0.0]

    if len(indices_with_class) > max_count:
        selected_with = random.sample(indices_with_class, max_count)
    else:
        selected_with = indices_with_class

    final_indices = selected_with + indices_without_class
    reduced_paths = [paths[i] for i in final_indices]
    reduced_labels = [labels[i] for i in final_indices]

    return reduced_paths, reduced_labels


def compute_class_weights(labels):
    labels_arr = np.array(labels)
    pos_counts = labels_arr.sum(axis=0)
    neg_counts = len(labels_arr) - pos_counts
    weights = neg_counts / (pos_counts + 1e-8)  # inverse prevalence
    return torch.tensor(weights, dtype=torch.float32)




def train_multilabel_model(model, train_loader, val_loader, num_epochs,
                           criterion, optimizer, scheduler, device, class_names):
    threshold = 0.35
    nsr_index = CLASS2IDX["59118001"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, demo, y in train_loader:
            x, demo, y = x.to(device), demo.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, demo)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss, all_y, all_preds = 0.0, [], []
        with torch.no_grad():
            for x, demo, y in val_loader:
                x, demo, y = x.to(device), demo.to(device), y.to(device)
                output = model(x, demo)
                loss = criterion(output, y)
                val_loss += loss.item()
                pred = (torch.sigmoid(output) > threshold).float()

                # === Fallback do NSR ===
                for i in range(pred.shape[0]):
                    if pred[i].sum() == 0:
                        pred[i, nsr_index] = 1.0

                all_y.append(y.cpu())
                all_preds.append(pred.cpu())

        avg_val_loss = val_loss / len(val_loader)
        all_y = torch.cat(all_y).numpy()
        all_preds = torch.cat(all_preds).numpy()

        f1 = f1_score(all_y, all_preds, average=None, zero_division=0)
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        for i, name in enumerate(class_names):
            print(f"{name:<6} | F1: {f1[i]:.3f}")

        print(f"\nMacro F1: {f1_score(all_y, all_preds, average='macro'):.4f}")
        scheduler.step(avg_val_loss)

    torch.save(model.state_dict(), "ecg_resnet1d_best_multilabel.pt")





# === Użycie nowej funkcji w main ===
if __name__ == "__main__":
    print("\U0001F4E6 Ładowanie ścieżek i multi-label etykiet...")
    paths, labels = load_multilabel_paths_and_labels_filtered()

    for code in ["426783006", "59118001", "270492004"]:
        paths, labels = reduce_class_occurrences(paths, labels, target_code=code, max_count=MAX_SAMPLES, class2idx=CLASS2IDX)

    plot_class_distribution(labels, CLASS_NAMES,
                            title="Multi-label – histogram klas",
                            filename="hist_multi_label.png")

    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths, labels, test_size=VAL_RATIO, random_state=SEED
    )

    train_ds = ECGDatasetMultiLabelWithDemo(paths_train, labels_train, seg_len=SEG_LEN)
    val_ds = ECGDatasetMultiLabelWithDemo(paths_val, labels_val, seg_len=SEG_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_skip_none)

    model = SE_ResNet1D(num_classes=len(CLASS_NAMES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4)

    pos_weights = compute_class_weights(labels_train).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    #criterion = nn.BCEWithLogitsLoss()

    train_multilabel_model(
        model, train_loader, val_loader,
        NUM_EPOCHS, criterion, optimizer, scheduler,
        DEVICE, class_names=CLASS_NAMES
    )



