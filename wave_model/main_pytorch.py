

import glob
import wfdb
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import  ConfusionMatrixDisplay


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Preferowane leady
PREFERRED_LEADS = [ "II","ii"  ]
# Ścieżki do baz danych
LUDB_PATH = "ludb/data/"
MODEL_PATH = "unet_ecg_fold5.h5"  # Model UNet
TARGET_FS = 500
WINDOW_SIZE = 2000
BAD_PATIENTS = [7, 34, 90,  95, 104, 111]
BAD_PATIENTS_II = []
BAD_PATIENTS_III = [95,104,111]

# Mapowanie symboli na klasy (0=none, 1=P, 2=QRS, 3=T)
WAVE_MAP = {'p': 1, 'N': 2, 't': 3}  # 0 = none
TOLERANCE = 150
EPOCHS = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_signal(signal):
    """Dodaje realistyczne zakłócenia do sygnału EKG."""
    L = len(signal)

    # 1. Szum Gaussowski (małe zakłócenia elektryczne)
    noise = np.random.normal(0, 0.01, L)

    # 2. Dryft bazowy (symulacja oddychania, niskoczęstotliwościowy trend)
    drift = 0.05 * np.sin(np.linspace(0, 2 * np.pi, L))

    # 3. Zakłócenia od mikro skurczów mięśni (szybkie, losowe zmiany)
    muscle_noise = 0.02 * np.random.randn(L) * np.sin(np.linspace(0, 50 * np.pi, L))

    # 4. Zakłócenia elektryczne 50Hz (lekkie zakłócenia sieci elektrycznej)
    electric_noise = 0.01 * np.sin(2 * np.pi * 50 * np.linspace(0, 1, L))

    # Skalujemy każdą perturbację losowym współczynnikiem w zakresie 0.5x - 1.5x
    noise *= np.random.uniform(0.5, 1.5)
    drift *= np.random.uniform(0.5, 1.5)
    muscle_noise *= np.random.uniform(0.5, 1.5)
    electric_noise *= np.random.uniform(0.5, 1.5)

    # Sumujemy wszystkie zakłócenia razem
    augmented_signal = signal + noise + drift + muscle_noise + electric_noise

    return augmented_signal






def select_best_lead(record):
    if record.p_signal is None or not hasattr(record, 'sig_name'):
        return None
    for lead in PREFERRED_LEADS:
        if lead in record.sig_name:
            print(f"[DEBUG] Wybrano lead: {lead}")

            return record.p_signal[:, record.sig_name.index(lead)]
    print("[DEBUG] Żaden preferowany lead nie został znaleziony.")
    return None

def find_annotation_file(record_name):
    possible_files = glob.glob(os.path.join(LUDB_PATH, record_name + ".*"))
    for lead in PREFERRED_LEADS:
        for file in possible_files:
            if file.endswith(f".{lead}"):
                print(f"[DEBUG] Znaleziono plik adnotacji: {file}")
                return file
    print(f"[DEBUG] Brak pliku adnotacji dla rekordu {record_name}")
    return None


def load_ecg(record_name):
    """Wczytuje sygnał EKG i adnotacje dla danego rekordu."""
    annotation_file = find_annotation_file(record_name)
    if annotation_file is None:
        return None, None

    record_path = os.path.join(LUDB_PATH, record_name)
    record = wfdb.rdrecord(record_path)
    ext = annotation_file.split('.')[-1]
    annotation = wfdb.rdann(annotation_file[:-len(ext)-1], extension=ext)

    best_lead = None
    for lead in PREFERRED_LEADS:
        if lead in record.sig_name:
            best_lead = record.p_signal[:, record.sig_name.index(lead)]
            break

    if best_lead is None:
        return None, None

    return best_lead, annotation  # Zwracamy rzeczywisty sygnał, a nie cały obiekt `Record`



def load_all_records_ludb():
    data_list = []
    for record_id in range(1, 201):
        if record_id in BAD_PATIENTS_III:
            print(f"[DEBUG] Pomijam pacjenta {record_id}")
            continue
        record_name = str(record_id)
        rec, ann = load_ecg(record_name)
        if rec is not None and ann is not None:
            best_lead = rec  # Już jest numpy.ndarray
            if best_lead is not None:
                data_list.append((best_lead, ann))
    print(f"[DEBUG] Łącznie wczytano {len(data_list)} rekordów")
    return data_list

def create_label_array(signal, annotation):
    L = len(signal)
    labels = np.zeros(L, dtype=int)
    i = 0
    while i < len(annotation.symbol):
        # Szukamy sekwencji: '(' <symbol> ')'
        if (annotation.symbol[i] == '(' and
                i+2 < len(annotation.symbol) and
                annotation.symbol[i+2] == ')' and
                annotation.symbol[i+1] in WAVE_MAP):
            wave_class = WAVE_MAP[annotation.symbol[i+1]]
            start_idx = annotation.sample[i]
            end_idx   = annotation.sample[i+2]
            start_idx = max(0, start_idx)
            end_idx   = min(L-1, end_idx)
            labels[start_idx:end_idx+1] = wave_class
            i += 3
        else:
            i += 1
    binc = np.bincount(labels, minlength=4)
    print(f"[DEBUG] Rozkład etykiet: none={binc[0]}, P={binc[1]}, QRS={binc[2]}, T={binc[3]}")
    return labels

def generate_training_fragments(signal, labels, num_fragments=5):
    L = len(signal)
    start_min = 1000
    start_max = L - 1000 - WINDOW_SIZE
    if start_max <= start_min:
        return [], [], [], []

    X_segments = []
    Y_segments = []
    X_aug_segments = []
    Y_aug_segments = []

    for _ in range(num_fragments):
        start_idx = randint(start_min, start_max)
        end_idx = start_idx + WINDOW_SIZE

        X_seg = signal[start_idx:end_idx].copy()
        Y_seg = labels[start_idx:end_idx].copy()

        # Normalizacja
        X_seg = (X_seg - np.mean(X_seg)) / (np.std(X_seg) + 1e-8)

        # Tworzymy wersję z zakłóceniami
        X_aug = augment_signal(X_seg)

        X_segments.append(X_seg)
        Y_segments.append(Y_seg)

        X_aug_segments.append(X_aug)
        Y_aug_segments.append(Y_seg)  # Adnotacje pozostają bez zmian!

    return X_segments, Y_segments, X_aug_segments, Y_aug_segments




# === U-Net 1D Model in PyTorch ===
class UNet1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=4):
        super(UNet1D, self).__init__()

        def CBR(in_ch, out_ch, k=9):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            )

        self.enc1 = CBR(input_channels, 4)
        self.pool1 = nn.MaxPool1d(2, padding=0)
        self.enc2 = CBR(4, 8)
        self.pool2 = nn.MaxPool1d(2, padding=0)
        self.enc3 = CBR(8, 16)
        self.pool3 = nn.MaxPool1d(2, padding=0)
        self.enc4 = CBR(16, 32)
        self.pool4 = nn.MaxPool1d(2, padding=0)
        self.bottleneck = CBR(32, 64)

        self.up4 = nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3, output_padding=0)
        self.dec4 = CBR(64, 32)
        self.up3 = nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2, padding=3, output_padding=0)
        self.dec3 = CBR(32, 16)
        self.up2 = nn.ConvTranspose1d(16, 8, kernel_size=8, stride=2, padding=3, output_padding=0)
        self.dec2 = CBR(16, 8)
        self.up1 = nn.ConvTranspose1d(8, 4, kernel_size=8, stride=2, padding=3, output_padding=0)
        self.dec1 = CBR(8, 4)

        self.out_conv = nn.Conv1d(4, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)
        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        u4 = self.pad_to_match(u4, c4)
        d4 = self.dec4(torch.cat([u4, c4], dim=1))

        u3 = self.up3(d4)
        u3 = self.pad_to_match(u3, c3)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3)
        u2 = self.pad_to_match(u2, c2)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        u1 = self.pad_to_match(u1, c1)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        out = self.out_conv(d1)
        return out.permute(0, 2, 1)  # (B, T, C)

    def pad_to_match(self, x, ref):
        diff = ref.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, (0, diff))
        return x


# === Custom Dataset ===
class ECGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def plot_confusion_matrix_samples(model, X, Y_onehot):
    preds = model.predict(X)
    pred_labels = np.argmax(preds, axis=-1).flatten()
    true_labels = np.argmax(Y_onehot, axis=-1).flatten()

    cm = confusion_matrix(true_labels, pred_labels, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["none","P","QRS","T"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (sample-level)")
    plt.show()

    print("\n[DEBUG] Classification Report (sample-level):")
    print(classification_report(true_labels, pred_labels,
                                labels=[0,1,2,3],
                                target_names=["none","P","QRS","T"]))


# ================== DODATKOWE FUNKCJE ONSET/OFFSET ==================
def extract_segments_from_prediction(pred_labels):
    segments = []
    current_class = pred_labels[0]
    start = 0
    for i in range(1, len(pred_labels)):
        if pred_labels[i] != current_class:
            if current_class != 0:
                segments.append((start, i - 1, current_class))
            current_class = pred_labels[i]
            start = i
    if current_class != 0:
        segments.append((start, len(pred_labels) - 1, current_class))
    return segments

def evaluate_onset_offset(true_segments, pred_segments, tolerance=TOLERANCE):
    used_pred = set()
    TP = 0
    for (true_start, true_end, true_class) in true_segments:
        found_match = False
        for j, (pred_start, pred_end, pred_class) in enumerate(pred_segments):
            if j in used_pred:
                continue
            if pred_class == true_class:
                onset_diff = abs(pred_start - true_start)
                offset_diff = abs(pred_end - true_end)
                if (onset_diff <= tolerance) and (offset_diff <= tolerance):
                    TP += 1
                    used_pred.add(j)
                    found_match = True
                    break
        # brak dopasowania => FN (liczony niżej)
    FP = len(pred_segments) - len(used_pred)
    FN = len(true_segments) - TP
    return TP, FP, FN

def evaluate_onset_offset_for_dataset(model, X, Y, tolerance=TOLERANCE):

    preds = predict_softmax(model, X, device)
    total_TP = 0
    total_FP = 0
    total_FN = 0
    for i in range(len(X)):
        pred_labels = np.argmax(preds[i], axis=-1)  # (2000,)
        true_labels = np.argmax(Y[i], axis=-1)      # (2000,)

        pred_segments = extract_segments_from_prediction(pred_labels)
        true_segments = extract_segments_from_prediction(true_labels)

        TP, FP, FN = evaluate_onset_offset(true_segments, pred_segments, tolerance=TOLERANCE)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    return total_TP, total_FP, total_FN


def validate_annotations_all_records():
    broken_records = []

    for record_id in range(1, 201):
        if record_id in BAD_PATIENTS_III:
            continue

        record_name = str(record_id)
        signal, annotation = load_ecg(record_name)
        if signal is None or annotation is None:
            print(f"[DEBUG] Pominięto {record_name}")
            continue

        symbols = annotation.symbol
        errors = []
        stack = []

        for i, sym in enumerate(symbols):
            if sym == '(':
                if stack:
                    errors.append(f"[{record_name}] Zagnieżdżony nawias '(' przy indexie {i}")
                stack.append(i)
            elif sym == ')':
                if not stack:
                    errors.append(f"[{record_name}] Samotny nawias ')' bez odpowiadającego '(' przy indexie {i}")
                else:
                    open_idx = stack.pop()
                    # Zakazane: nawias domknięty natychmiast po wcześniejszym
                    if open_idx + 1 == i:
                        errors.append(f"[{record_name}] Pusta para nawiasów '()' przy indexie {open_idx}")

        # Jeśli po przejściu coś zostało w stosie, to otwarte nawiasy
        if stack:
            for idx in stack:
                errors.append(f"[{record_name}] Niezamknięty nawias '(' przy indexie {idx}")

        if errors:
            print(f"\n❌ Błędy w rekordzie {record_name}:")
            for err in errors:
                print("  " + err)
            broken_records.append(record_name)
        else:
            print(f"[DEBUG] ✅ Rekord {record_name} – poprawna sekwencja nawiasów")

    print("\n=== Podsumowanie ===")
    print(f"Niepoprawne rekordy: {broken_records}")
    return broken_records




def predict_softmax(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device)  # (B, 1, T)
        logits = model(X_tensor)  # (B, T, C)
        preds = F.softmax(logits, dim=-1).cpu().numpy()
    return preds


# === Training + CV ===
def train_model():
    X = np.load("X_total.npy")
    Y = np.load("Y_total.npy")



    dataset = ECGDataset(X, Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        print(f"[FOLD {fold}] Training...")
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        model = UNet1D()
        model = model.cuda() if torch.cuda.is_available() else model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(1, EPOCHS+1):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out.reshape(-1, 4), yb.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out.reshape(-1, 4), yb.view(-1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"unet_ecg_fold{fold}.pt")
                print("✅ Model saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹ Early stopping")
                    break

        # Ewaluacja onset/offset
        X_val_np = X[val_idx]
        Y_val_np = Y[val_idx]
        preds = predict_softmax(model, X_val_np, device)

        TP, FP, FN = evaluate_onset_offset_for_dataset(model, X_val_np, np.eye(4)[Y_val_np])
        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print(f"[FOLD {fold}] Onset/Offset: TP={TP}, FP={FP}, FN={FN}")
        print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        y_true = Y_val_np.flatten()
        y_pred = np.argmax(preds, axis=-1).flatten()

        print(classification_report(y_true, y_pred, target_names=["none", "P", "QRS", "T"]))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["none", "P", "QRS", "T"], cmap="Blues")
        plt.title(f"Confusion Matrix Fold {fold}")
        plt.show()


if __name__ == "__main__":
    train_model()


