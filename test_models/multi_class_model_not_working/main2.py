import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.signal import resample, butter, sosfiltfilt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from torch.multiprocessing import freeze_support

#########################################
# USTAWIENIA I DEFINICJE ŚCIEŻEK
#########################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'C:\Users\msztu\Documents\EKG_DB_2\challenge2020_data\training'
EXCLUDED_DIRS = ["ptb", "st_petersburg_incart"]  # Katalogi do pominięcia
SAVE_DIR = os.path.join(BASE_DIR, 'runtime')
x_path = os.path.join(BASE_DIR, 'X.npy')
y_path = os.path.join(BASE_DIR, 'Y.npy')

#########################################
# PARAMETRY SYGNAŁU I DEFINICJA KLAS
#########################################
FS_TARGET = 500           # docelowa częstotliwość w Hz
WINDOW_SEC = 10           # długość okna – 10 sekund
SEG_LENGTH = FS_TARGET * WINDOW_SEC  # 5000 próbek
LEADS = 12                # liczba odprowadzeń
MIN_SEC = 8               # minimalna długość sygnału (8 s = 4000 próbek)

# Poprawiona definicja klas – zgodnie z Table 3 z Challenge 2020
# Każdy kod (klucz) to SNOMED CT, a wartość to skrót diagnozy
SNOMED_IMAGE_CLASSES = {
    '270492004': 'IAVB',       # 1st degree AV block
    '164889003': 'AF',         # Atrial fibrillation
    '164890007': 'AFL',        # Atrial flutter
    '426627000': 'Brady',      # Bradycardia
    '713427006': 'CRBBB',      # Complete right bundle branch block
    '713426002': 'IRBBB',      # Incomplete right bundle branch block
    '445118002': 'LAnFB',      # Left anterior fascicular block
    '39732003':  'LAD',        # Left axis deviation
    '164909002': 'LBBB',       # Left bundle branch block
    '251146004': 'LQRSV',      # Low QRS voltages
    '698252002': 'NSIVCB',     # Nonspecific intraventricular conduction disorder
    '10370003':  'PR',         # Pacing rhythm
    '284470004': 'PAC',        # Premature atrial contraction
    '427172004': 'PVC',        # Premature ventricular contractions
    '164947007': 'LPR',        # Prolonged PR interval
    '111975006': 'LQT',        # Prolonged QT interval
    '164917005': 'QAb',        # Q wave abnormal
    '47665007':  'RAD',        # Right axis deviation
    '59118001':  'RBBB',       # Right bundle branch block
    '427393009': 'SA',         # Sinus arrhythmia
    '426177001': 'SB',         # Sinus bradycardia
    '426783006': 'NSR',        # Normal sinus rhythm
    '427084000': 'STach',      # Sinus tachycardia
    '63593006':  'SVPB',       # Supraventricular premature beats
    '164934002': 'TAb',        # T wave abnormal
    '59931005':  'TInv',       # T wave inversion
    '17338001':  'VPB'         # Ventricular premature beats
}

#########################################
# FUNKCJE FILTROWANIA SYGNAŁU
#########################################
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS_TARGET, order=4):
    """
    Stosuje filtr pasmowo-przepustowy Butterwortha (sekcje SOS) do usuwania
    zakłóceń mięśniowych, elektrycznych i oddechowych z sygnału ECG.
    Referencje: Ioffe & Szegedy (2015) dla BatchNorm; He et al. (2016) dla residuals.
    """
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    y = sosfiltfilt(sos, data, axis=1)
    return y

#########################################
# FUNKCJE PRZETWARZANIA DANYCH
#########################################
def list_hea_files(data_dir):
    """
    Rekurencyjnie przeszukuje folder DATA_DIR i zwraca ścieżki do plików .hea
    (bez rozszerzenia), pomijając katalogi z listy EXCLUDED_DIRS.
    """
    paths = []
    for root, _, files in os.walk(data_dir):
        # Jeśli w ścieżce znajduje się któryś z wykluczonych katalogów, pomijamy
        if any(ex_dir in root for ex_dir in EXCLUDED_DIRS):
            continue

        for f in files:
            if f.endswith('.hea'):
                paths.append(os.path.join(root, f[:-4]))
    return paths

def load_snomed_codes(hea_path):
    """
    Wczytuje kody SNOMED z pliku .hea.
    Normalizuje linię zawierającą etykiety (np. "# Dx:" lub "#Dx:") i zwraca listę
    tylko tych kodów, które znajdują się w SNOMED_IMAGE_CLASSES.
    """
    codes = []
    with open(hea_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_norm = line.strip().replace('# Dx:', '#Dx:')
        if line_norm.startswith('#Dx:'):
            code_str = line_norm.split(':', 1)[1].strip()
            for c in code_str.split(','):
                c = c.strip()
                if c in SNOMED_IMAGE_CLASSES:
                    codes.append(c)
            break
    return codes

def process_signal(signal, seg_length=SEG_LENGTH, fs_target=FS_TARGET):
    """
    Dzieli sygnał na segmenty o długości 10 sekund (5000 próbek).
    - Jeśli sygnał jest krótszy niż 10 sekund, ale dłuższy niż 8 sekund, dopełnia go paddingiem.
    - Jeśli sygnał jest krótszy niż 8 sekund, odrzuca.
    - Jeśli sygnał jest dłuższy niż 10 sekund, dzieli go na pełne okna, a ostatni segment dopełnia.
    """
    n_samples = signal.shape[1]
    segments = []
    min_samples = MIN_SEC * fs_target  # 4000 próbek
    if n_samples < min_samples:
        return segments
    if n_samples < seg_length:
        padded = np.pad(signal, ((0, 0), (0, seg_length - n_samples)), mode='constant')
        segments.append(padded)
        return segments
    num_full = n_samples // seg_length
    for i in range(num_full):
        seg = signal[:, i*seg_length:(i+1)*seg_length]
        segments.append(seg)
    remainder = n_samples % seg_length
    if remainder >= min_samples:
        last_seg = signal[:, num_full*seg_length:]
        last_seg = np.pad(last_seg, ((0, 0), (0, seg_length - remainder)), mode='constant')
        segments.append(last_seg)
    return segments

def prepare_and_save_data_and_labels(data_dir, fs_target=FS_TARGET, window_sec=WINDOW_SEC, leads=LEADS):
    seg_length = fs_target * window_sec  # 10 sekund = 5000 próbek
    max_duration = 100 * fs_target         # 100 sekund
    metadata = []
    segments = []
    labels_list = []
    file_counter = 0
    hea_files = list_hea_files(data_dir)

    classes = list(SNOMED_IMAGE_CLASSES.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])

    for base_path in hea_files:
        hea_path = base_path + '.hea'
        mat_path = base_path + '.mat'
        if not (os.path.exists(hea_path) and os.path.exists(mat_path)):
            continue

        codes = load_snomed_codes(hea_path)
        if not codes:
            continue

        with open(hea_path, 'r') as f:
            header_line = f.readline().split()
        if len(header_line) < 4:
            continue
        try:
            n_leads = int(header_line[1])
            fs_orig = float(header_line[2])
        except:
            continue
        if n_leads != leads:
            continue

        try:
            mat_data = loadmat(str(mat_path))
        except Exception:
            continue
        if 'val' not in mat_data:
            continue
        signal = mat_data['val']
        if fs_orig != fs_target:
            target_len = int(signal.shape[1] * fs_target / fs_orig)
            signal = resample(signal, target_len, axis=1)

        # Sprawdź, czy sygnał ma maksymalnie 20 sekund
        if signal.shape[1] > max_duration:
            continue

        signal = butter_bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=fs_target, order=4)
        segments_batch = process_signal(signal, seg_length, fs_target)
        if not segments_batch:
            continue
        y = mlb.transform([list(set(codes))])[0]
        for seg in segments_batch:
            segments.append(seg.astype(np.float32))
            labels_list.append(y.astype(np.float32))
            metadata.append({'file': f'seg_{file_counter}.npy', 'labels': list(np.where(y == 1)[0])})
            file_counter += 1

    # Zapis wszystkich segmentów i etykiet jako pliki .npy
    np.save(os.path.join(BASE_DIR, 'X.npy'), np.stack(segments))
    np.save(os.path.join(BASE_DIR, 'Y.npy'), np.stack(labels_list))

    return file_counter

#########################################
# ŁADOWANIE DANYCH DO RAM (cały zbiór)
#########################################
def load_data_into_memory():
    """
    Wczytuje pliki X.npy i Y.npy do pamięci RAM.
    """
    X_path = os.path.join(BASE_DIR, 'X.npy')
    Y_path = os.path.join(BASE_DIR, 'Y.npy')
    X = np.load(X_path)
    Y = np.load(Y_path)
    return X, Y

#########################################
# DEFINICJA PYTORCH DATASET – z danymi w RAM
#########################################
class ECGMemoryDataset(Dataset):
    def __init__(self, X, Y, augment=False):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        # Opcjonalna augmentacja (np. dodanie szumu Gaussian) – tylko dla zbioru treningowego
        if self.augment:
            noise = np.random.normal(0, 0.01, size=x.shape)
            x = x + noise
        y = self.Y[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

#########################################
# DEFINICJA MODELU CNN + LSTM + ATTENTION
#########################################
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, time, hidden_dim)
        attn_weights = torch.tanh(self.attn(lstm_out))   # (batch, time, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # normalizacja po czasie
        context = torch.sum(attn_weights * lstm_out, dim=1)  # agregacja kontekstu
        return context

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, in_channels=LEADS, num_classes=None):
        super(CNN_LSTM_Attention, self).__init__()
        if num_classes is None:
            num_classes = len(SNOMED_IMAGE_CLASSES)
        # Bloki konwolucyjne – ekstrakcja cech z sygnału (odnosi się do podejścia CNN w literaturze)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Warstwa LSTM – modelowanie sekwencji czasowej (zgodne z pracami nad LSTM w ECG)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1,
                            batch_first=True, bidirectional=True)
        # Mechanizm uwagi – agregacja informacji z LSTM (podobnie jak w Transformerach)
        self.attention = Attention(hidden_dim=512)  # 256 * 2 (bidirectional)
        # Warstwy w pełni połączone – klasyfikacja wieloetykietowa z wyjściem Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Wejście: (batch, 12, 5000)
        x = self.conv1(x)   # -> (batch, 64, 2500)
        x = self.conv2(x)   # -> (batch, 128, 1250)
        x = self.conv3(x)   # -> (batch, 256, ~625)
        x = x.permute(0, 2, 1)  # zmiana wymiarów na (batch, time, features)
        lstm_out, _ = self.lstm(x)  # (batch, ~625, 512)
        context = self.attention(lstm_out)  # (batch, 512)
        out = self.fc(context)  # (batch, num_classes)
        return out

#########################################
# FUNKCJE TRENINGOWE I EWALUACYJNE
#########################################
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
    avg_loss = running_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return avg_loss, all_preds, all_targets

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())
    avg_loss = running_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return avg_loss, all_preds, all_targets

def print_metrics(y_true, y_pred, classes):
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[SNOMED_IMAGE_CLASSES[c] for c in classes], zero_division=0))
    for idx, cls in enumerate(classes):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        print(f"Confusion matrix for {SNOMED_IMAGE_CLASSES[cls]}:\n{cm}\n")

#########################################
# GŁÓWNY PROGRAM – TRENING, EWALUACJA, ZAPIS MODELU
#########################################
if __name__ == '__main__':
    freeze_support()

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("Przetwarzam dane i zapisuję segmenty na dysku...")
        num_segments = prepare_and_save_data_and_labels(DATA_DIR)
        print(f"Zapisano {num_segments} segmentów.")
    else:
        print("Dane już przygotowane.")



    print("Wczytuje wszystkie segmenty do pamięci RAM...")
    X = np.load(os.path.join(BASE_DIR, 'X.npy'))
    Y = np.load(os.path.join(BASE_DIR, 'Y.npy'))
    print(f"Kształt X: {X.shape}, Kształt Y: {Y.shape}")

    classes = list(SNOMED_IMAGE_CLASSES.keys())


    # Podział danych na zbiory: trening (80%), walidacja (10%) i test (10%)
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, train_size=0.8, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_dataset = ECGMemoryDataset(X[train_idx], Y[train_idx], augment=True)
    val_dataset = ECGMemoryDataset(X[val_idx], Y[val_idx], augment=False)
    test_dataset = ECGMemoryDataset(X[test_idx], Y[test_idx], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Inicjalizacja modelu
    num_classes = len(classes)
    model = CNN_LSTM_Attention(in_channels=LEADS, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_preds, train_targets = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print("  Train Metrics:")
        print_metrics(train_targets, train_preds, classes)
        print("  Val Metrics:")
        print_metrics(val_targets, val_preds, classes)
        # Zapis modelu po każdej epoce
        checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Model zapisany jako {checkpoint_path}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  *** Best model zapisany jako {best_model_path} ***")

    # Ewaluacja na zbiorze testowym
    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    print("\nTest Metrics:")
    print_metrics(test_targets, test_preds, classes)