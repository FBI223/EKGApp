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

#########################################
# USTAWIENIA I DEFINICJE ŚCIEŻEK
#########################################
# Folder, w którym znajduje się ten skrypt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ścieżka do danych – dostosuj, jeśli potrzeba
DATA_DIR = r'C:\Users\msztu\Documents\EKG_DB_2\challenge2020_data\training'
# Folder zapisu przygotowanych danych (utworzony w katalogu programu)
SAVE_DIR = os.path.join(BASE_DIR, 'preprocessed')
SEG_DIR = os.path.join(SAVE_DIR, 'segments')
METADATA_FILE = os.path.join(SAVE_DIR, 'metadata.csv')
os.makedirs(SEG_DIR, exist_ok=True)

#########################################
# PARAMETRY SYGNAŁU I KLAS
#########################################
FS_TARGET = 500           # docelowa częstotliwość (Hz)
WINDOW_SEC = 10           # długość okna (sekundy)
SEG_LENGTH = FS_TARGET * WINDOW_SEC  # 5000 próbek
LEADS = 12                # liczba odprowadzeń
MIN_SEC = 8               # minimalna długość sygnału (8 s = 4000 próbek)

# Słownik SNOMED – przetwarzamy tylko te kody
SNOMED_IMAGE_CLASSES = {
    '10370003': 'AVB_II',
    '111975006': 'AFL',
    '164889003': 'AF',
    '164890007': 'AF_RVR',
    '164909002': 'LBBB',
    '164917005': 'RBBB',
    '164934002': 'RAD',
    '164947007': 'LQT',
    '251146004': 'LQRSV',
    '270492004': 'IAVB',
    '284470004': 'PAC',
    '39732003':  'PR',
    '426177001': 'LAD',
    '426627000': 'Brady',
    '426783006': 'NSR',
    '427084000': 'SB',
    '427172004': 'SA',
    '427393009': 'STach',
    '445118002': 'LAnFB',
    '47665007':  'PJB',
    '59931005':  'LVH',
    '698252002': 'SVPB',
    '713426002': 'IRBBB',
    '713427006': 'CRBBB',
    '164931005': 'LPFB',
    '164884008': 'PVC',
    '251182009': 'NSIVCB'
}

#########################################
# FUNKCJE FILTROWANIA SYGNAŁU
#########################################
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS_TARGET, order=4):
    """
    Filtr pasmowo-przepustowy Butterwortha z wykorzystaniem drugorzędowych sekcji (sos)
    dla stabilności numerycznej. Usuwa zakłócenia mięśniowe, elektryczne i oddechowe.
    """
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    y = sosfiltfilt(sos, data, axis=1)
    return y

#########################################
# FUNKCJE PRZETWARZANIA DANYCH
#########################################
def list_hea_files(data_dir):
    """ Rekurencyjnie znajduje wszystkie pliki .hea i zwraca ich ścieżki (bez rozszerzenia). """
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.hea'):
                paths.append(os.path.join(root, f[:-4]))
    return paths

def load_snomed_codes(hea_path):
    """
    Wczytuje kody SNOMED z pliku .hea, normalizując linie aby wykryć zarówno "#Dx:" jak i "# Dx:".
    Zwraca listę kodów, pomijając te, których nie ma w SNOMED_IMAGE_CLASSES.
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
    Dzieli sygnał na segmenty 10‑sekundowe (5000 próbek).
      - Jeśli sygnał jest krótszy niż 10 s, ale >=8 s – dopełnia paddingiem.
      - Jeśli sygnał jest krótszy niż 8 s, odrzuca.
      - Jeśli sygnał jest dłuższy, dzieli go na pełne okna, a ostatni segment (jeśli >=8 s) dopełnia.
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

def prepare_and_save_data(data_dir, fs_target=FS_TARGET, window_sec=WINDOW_SEC, leads=LEADS):
    """
    Przetwarza wszystkie pliki w folderze DATA_DIR:
      - Wczytuje pliki .hea i .mat.
      - Resampluje sygnał do fs_target.
      - Filtruje sygnał (bandpass 0.5–40 Hz).
      - Dzieli sygnał na 10‑sekundowe segmenty.
      - Wczytuje kody SNOMED (tylko te zdefiniowane w SNOMED_IMAGE_CLASSES).
      - Zapisuje każdy segment jako oddzielny plik .npy w SEG_DIR.
      - Zapisuje metadane (ścieżka segmentu i etykiety) w METADATA_FILE.
    """
    seg_length = fs_target * window_sec
    metadata = []
    file_counter = 0
    hea_files = list_hea_files(data_dir)
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
        signal = mat_data['val']  # kształt: [leads, n_samples]
        if fs_orig != fs_target:
            target_len = int(signal.shape[1] * fs_target / fs_orig)
            signal = resample(signal, target_len, axis=1)
        # Filtrowanie sygnału
        signal = butter_bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=fs_target, order=4)
        segments = process_signal(signal, seg_length, fs_target)
        if not segments:
            continue
        unique_codes = list(set(codes))
        for seg in segments:
            seg_filename = f"seg_{file_counter}.npy"
            seg_path = os.path.join(SEG_DIR, seg_filename)
            np.save(seg_path, seg.astype(np.float32))
            metadata.append({
                'segment_path': seg_path,
                'labels': unique_codes
            })
            file_counter += 1
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_FILE, index=False)
    return metadata

#########################################
# DEFINICJA PYTORCH DATASET (ŁADOWANIE DANYCH Z DYSKU)
#########################################
class ECGDiskDataset(Dataset):
    def __init__(self, metadata_file, mlb):
        self.df = pd.read_csv(metadata_file)
        # Konwersja string reprezentujący listę etykiet na listę
        self.df['labels'] = self.df['labels'].apply(eval)
        self.mlb = mlb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seg_path = self.df.iloc[idx]['segment_path']
        x = np.load(seg_path)
        labels = self.df.iloc[idx]['labels']
        y = self.mlb.transform([labels])[0]
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
        attn_weights = torch.tanh(self.attn(lstm_out))  # (batch, time, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, time, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        return context

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, in_channels=LEADS, num_classes=None):
        super(CNN_LSTM_Attention, self).__init__()
        # Upewnij się, że num_classes jest określone; jeśli nie, ustaw domyślnie
        if num_classes is None:
            num_classes = len(SNOMED_IMAGE_CLASSES)
        # Bloki konwolucyjne
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
        # Po trzech blokach, przy wejściu 5000 próbek, zostaje około 5000/8 ≈ 625 kroków czasowych.
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1,
                            batch_first=True, bidirectional=True)
        # Używamy mechanizmu uwagi na wyjściu LSTM
        self.attention = Attention(hidden_dim=256*2)  # bidirectional → 512 wymiarów
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Wieloetykietowa klasyfikacja
        )

    def forward(self, x):
        # x: (batch, 12, 5000)
        x = self.conv1(x)   # (batch, 64, 2500)
        x = self.conv2(x)   # (batch, 128, 1250)
        x = self.conv3(x)   # (batch, 256, ~625)
        x = x.permute(0, 2, 1)  # (batch, time, features) -> (batch, ~625, 256)
        lstm_out, _ = self.lstm(x)  # (batch, ~625, 512)
        context = self.attention(lstm_out)  # (batch, 512)
        out = self.fc(context)  # (batch, num_classes)
        return out

#########################################
# TRAINING PIPELINE
#########################################
if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()

    # Jeśli metadane nie istnieją, przygotuj dane i zapisz segmenty
    if not os.path.exists(METADATA_FILE):
        print("Przetwarzam dane i zapisuję segmenty na dysku...")
        metadata = prepare_and_save_data(DATA_DIR)
        print(f"Zapisano {len(metadata)} segmentów.")
    else:
        print("Dane już przygotowane. Wczytuję metadata.")

    # Przygotowanie MultiLabelBinarizer
    classes = list(SNOMED_IMAGE_CLASSES.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])

    # Tworzenie datasetu i DataLoadera
    dataset = ECGDiskDataset(METADATA_FILE, mlb)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Inicjalizacja modelu
    num_classes = len(classes)
    model = CNN_LSTM_Attention(in_channels=LEADS, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    num_epochs = 10
    for epoch in range( num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
