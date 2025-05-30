from collections import Counter

import pywt
import wfdb
from scipy.signal import resample
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,  multilabel_confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from model5 import SE_ResNet1D
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score , f1_score


# === Ustawienia ===
BATCH_SIZE = 64
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


MAX_SAMPLES=6000
LR=1e-3
TARGET_FS = 500


LEAD_IDX = 1
SEG_DUR = 10
SEG_LEN = TARGET_FS * SEG_DUR  # 5000 samples
AGE_MEAN = 60.0

BASE_PATH = "/home/msztu223/PycharmProjects/ECG_PROJ/databases/challenge2020_data/training"
SUBSETS = ['cpsc_2018', 'cpsc_2018_extra' , 'ptb-xl' , 'georgia']




CPSC_CODES  = [
    "164934002",  # AF
    "428750005",  # SVT
    "426783006",  # IAVB
    "426177001",  # NSR
    "164873001",  # STE
    "427084000",  # RBBB
    "284470004",  # PAC
    "164889003",  # LBBB
    "270492004",  # PVC
    "164884008",  # STD
]


CLASS_NAMES= [
    "AF", "SVT", "IAVB", "NSR", "STE",
    "RBBB", "PAC", "LBBB", "PVC", "STD"
]


# === Mapowanie kodów równoważnych (używane tylko przy normalizacji etykiet) ===
CODE_EQUIV = {
    "17338001": "284470004",   # SVPB → PAC
    "59118001": "270492004",   # VPB → PVC
    "164890007": "164889003",  # CRBBB → LBBB
    "39732003": "426177001",   # Sinus rhythm → NSR
    "47665007": "284470004",   # Atrial premature beat → PAC
    "111975006": "164873001",  # STEMI → STE
    "733534002": "164884008",  # STD MI → STD
}

THRESHOLD = 0.5
AF_THRESHOLD = 0.3
NSR_IDX = 3
AF_IDX = 0




# === Progi predykcji per klasa ===
THRESHOLDS = [AF_THRESHOLD if i == AF_IDX else THRESHOLD for i in range(len(CLASS_NAMES))]

# === Mapowanie kodu do indeksu (do użycia w wektorach multi-hot)
CLASS2IDX = {code: i for i, code in enumerate(CPSC_CODES)}


def oversample_minority(paths, labels, class_code, target_count, class2idx):
    idx = class2idx[class_code]
    minority_indices = [i for i in range(len(labels)) if labels[i][idx] == 1]
    current_count = len(minority_indices)
    if current_count == 0 or current_count >= target_count:
        return paths, labels

    repeat_times = target_count // current_count
    extra_paths = [paths[i] for i in minority_indices] * repeat_times
    extra_labels = [labels[i] for i in minority_indices] * repeat_times

    return paths + extra_paths, labels + extra_labels



def reduce_multiple_classes(paths, labels, class2idx, limits: dict):
    for code, max_count in limits.items():
        paths, labels = reduce_class_occurrences(paths, labels, target_code=code, max_count=max_count, class2idx=class2idx)
    return paths, labels

def optimize_thresholds(y_true, y_probs):
    best_thresh = []
    for i in range(y_true.shape[1]):
        scores = [(t, f1_score(y_true[:, i], (y_probs[:, i] > t).astype(int))) for t in np.arange(0.1, 0.9, 0.05)]
        best_thresh.append(max(scores, key=lambda x: x[1])[0])
    return best_thresh

def reduce_class_occurrences(paths, labels, target_code, max_count, class2idx):

    idx = class2idx[target_code]
    all_indices = list(range(len(labels)))

    # Przykłady zawierające daną klasę
    matching = [i for i in all_indices if labels[i][idx] == 1.0]
    keep_matching = set(random.sample(matching, min(max_count, len(matching))))

    # Przykłady nie zawierające tej klasy lub te wybrane
    final_indices = [i for i in all_indices if labels[i][idx] != 1.0 or i in keep_matching]

    paths_reduced = [paths[i] for i in final_indices]
    labels_reduced = [labels[i] for i in final_indices]

    return paths_reduced, labels_reduced



def plot_multilabel_combination_histogram(labels, max_combinations=20):

    labels = np.array(labels)
    combos = [''.join(str(int(b)) for b in row) for row in labels]
    counter = Counter(combos)

    most_common = counter.most_common(max_combinations)
    keys = [x[0] for x in most_common]
    vals = [x[1] for x in most_common]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=90)
    plt.xlabel("Kombinacja etykiet (binarnie)")
    plt.ylabel("Liczność")
    plt.title(f"Top {max_combinations} kombinacji multi-label (z {2 ** labels.shape[1]} możliwych)")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("multilabel_combination_histogram.png")
    plt.show()


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


# === NSR fallback ===
def apply_nsr_fallback(pred):
    fallback_pred = pred.clone()
    empty = fallback_pred.sum(dim=1) == 0
    fallback_pred[empty, NSR_IDX] = 1
    return fallback_pred

def augment_underrepresented(x, y):
    if y[CLASS2IDX["164884008"]] == 1 or y[CLASS2IDX["284470004"]] == 1:
        x_aug = x + np.random.normal(0, 0.01, size=x.shape)
        return x_aug, y
    return x, y


class SignLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        sign = torch.where(
            torch.abs(targets - probs) < 0.5,
            targets - 2 * probs * targets + probs ** 2,
            torch.ones_like(targets)
        )
        return (sign * self.bce(inputs, targets)).mean()



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-BCE)
        focal = ((1 - pt) ** self.gamma) * BCE
        return focal.mean()





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






class ECGDatasetWithDemo(Dataset):
    def __init__(self, paths, labels, class_list, seg_len=5000, target_fs=500):
        self.paths = paths
        self.labels = labels
        self.class_list = class_list
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
            sig = self.normalize(sig)

            if np.isnan(sig).any() or not np.isfinite(sig).all():
                return None

            age, sex, _ = parse_header(path + '.hea')
            if not np.isfinite(age) or not np.isfinite(sex):
                age, sex = 60.0, 0.0

            x_tensor = torch.tensor(sig[1:2], dtype=torch.float32)  # lead II
            demo_tensor = torch.tensor([age, sex], dtype=torch.float32)
            y_tensor = torch.tensor(label, dtype=torch.float32)

            return x_tensor, demo_tensor, y_tensor

        except Exception as e:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# === Funkcja trenowania ===
def train_multi_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device, class_names):
    best_val_loss = float('inf')
    patience = 8
    delta = 1e-4
    patience_counter = 0

    train_losses, val_losses = [], []
    all_y_true, all_y_pred = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x, demo, y in train_loader:
            x, demo, y = x.to(device), demo.to(device), y.to(device)
            outputs = model(x, demo)  # jeśli używasz demografii
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for x, demo, y in val_loader:
                x, demo, y = x.to(device), demo.to(device), y.to(device)
                outputs = model(x, demo)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs)
                preds = torch.stack([
                    (preds[:, i] > THRESHOLDS[i]).int() for i in range(preds.shape[1])
                ], dim=1)
                preds = apply_nsr_fallback(preds)
                y_true.append(y.cpu())
                y_pred.append(preds.cpu())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        all_y_true = y_true
        all_y_pred = y_pred

        # === Early stopping + scheduler ===
        scheduler.step(val_loss + 1e-8)

        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "ecg_resnet1d_best_multilabel.pt")
        else:
            patience_counter += 1

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        if epoch == num_epochs - 1:
            thresholds_opt = optimize_thresholds(y_true, torch.cat([torch.sigmoid(o) for o in outputs]).cpu().numpy())
            print("🔧 Optymalne progi:", thresholds_opt)

        try:
            auroc = roc_auc_score(y_true, y_pred, average="macro")
            print(f"AUROC macro: {auroc:.4f}")
        except ValueError as e:
            print(f"AUROC computation failed: {e}")

        if patience_counter > patience:
            print(f"⏹️ Early stopping at epoch {epoch + 1}")
            break


    # Wykres strat
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_multilabel.png")
    plt.show()

    return model, all_y_true, all_y_pred












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




# === Trenowanie ===
def run_multilabel_training(paths, labels, class_names, device):


    paths_train, paths_val, labels_train, labels_val = train_test_split(paths, labels, test_size=TEST_RATIO, random_state=SEED)

    train_dataset = ECGDatasetWithDemo(paths_train, labels_train, class_list=CPSC_CODES, seg_len=SEG_LEN)
    val_dataset = ECGDatasetWithDemo(paths_val, labels_val, class_list=CPSC_CODES, seg_len=SEG_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_skip_none, num_workers=4, persistent_workers=True)
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_skip_none)

    # Wagi klas
    all_labels = np.array(labels)
    class_counts = np.sum(all_labels, axis=0)
    class_weights = np.median(class_counts) / (class_counts + 1e-6)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = SE_ResNet1D(num_classes=len(class_names)).to(device)
    #criterion = SignLoss(pos_weight=weights_tensor)
    criterion = FocalLoss(gamma=2, pos_weight=weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

    model, y_true, y_pred = train_multi_model(
        model, train_loader, val_loader, NUM_EPOCHS,
        criterion, optimizer, scheduler, device, class_names
    )

    # Macierz pomyłek
    cm = multilabel_confusion_matrix(y_true, y_pred)
    print("Multilabel Confusion Matrix:\n", cm)

    # Zapis modelu
    torch.save(model.state_dict(), "ecg_resnet1d_best_multilabel.pt")


def balance_classes(paths, labels, target_count, class2idx):
    for code in class2idx:
        current_count = int(np.sum(np.array(labels)[:, class2idx[code]]))
        if current_count < target_count:
            paths, labels = oversample_minority(paths, labels, code, target_count, class2idx)
        elif current_count > target_count:
            paths, labels = reduce_class_occurrences(paths, labels, target_code=code, max_count=target_count, class2idx=class2idx)
    return paths, labels




if __name__ == "__main__":
    print("📦 Ładowanie ścieżek i etykiet...")

    paths, labels = load_multilabel_paths_and_labels()

    # 🔁 Balansowanie klas
    paths, labels = balance_classes(paths, labels, target_count=5000, class2idx=CLASS2IDX)

    # 📊 Histogram kombinacji multi-label
    plot_multilabel_combination_histogram(labels, max_combinations=30)

    # 📊 Histogram liczby etykiet per próbka
    labels_arr = np.array(labels)
    label_sums = labels_arr.sum(axis=1)

    plt.figure(figsize=(6, 4))
    plt.hist(label_sums, bins=np.arange(0, 4) - 0.5, rwidth=0.6)
    plt.xticks([0, 1, 2])
    plt.xlabel("Liczba etykiet")
    plt.ylabel("Liczba próbek")
    plt.title("Rozkład liczby etykiet per próbka")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("label_count_per_sample.png")
    plt.show()

    # ▶️ Trening modelu
    run_multilabel_training(paths, labels, CLASS_NAMES, DEVICE)


