
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import wfdb
from scipy.signal import resample
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np



# === Settings ===
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FS = 500
SEG_DUR = 10
SEG_LEN = SEG_DUR * TARGET_FS
LEAD_IDX = 1
AGE_MEAN = 60.0

BATCH_SIZE = 64
NUM_EPOCHS = 50
VAL_RATIO = 0.15

BASE_PATH = "/home/msztu223/PycharmProjects/ECG_PROJ/databases/challenge2020_data/training"
SUBSETS = ['cpsc_2018', 'cpsc_2018_extra']

# === Selected classes ===
CPSC_CODES = [
    "59118001",   # NSR
    "164889003",  # LBBB
    "426783006",  # IAVB
    "429622005",  # SVT
    "270492004",  # PVC
    "164884008",  # TINV
    "284470004",  # STD
    "164909002",  # AF
    "428750005",  # PAC
    "164867002",  # ST
]
CLASS2IDX = {code: i for i, code in enumerate(CPSC_CODES)}
CLASS_NAMES = list(CLASS2IDX.keys())

# === Equivalent codes ===
CODE_EQUIV = {
    "17338001": "428750005",   # SVPB -> PAC
    "47665007": "428750005",   # atrial premature -> PAC
    "164890007": "164889003",  # CRBBB -> LBBB
    "39732003": "59118001",    # sinus rhythm -> NSR
    "164896001": "59118001",
    "733534002": "284470004",  # MI w STD
    "111975006": "164873001",  # STEMI -> STE
}


wavelet = 'bior2.6'
start_thresh_level = 4
threshold_scale = 0.3
threshold_mode = 'soft'



# === Parse header ===
def parse_header(header_path):
    age, sex, codes = None, None, []
    with open(header_path, 'r') as f:
        for line in f:
            if line.startswith('# Age:'):
                try:
                    age = float(line.split(':')[1].strip())
                    if not (0 <= age <= 110):
                        age = None
                except:
                    age = None
            elif line.startswith('# Sex:'):
                sex = 1.0 if 'female' in line.lower() else 0.0
            elif 'Dx:' in line:
                raw = line.split(':')[1].split(',')
                codes = [CODE_EQUIV.get(c.strip(), c.strip()) for c in raw]
    return age or AGE_MEAN, sex or 0.0, codes

# === Load multi-label dataset ===
def load_multi_label_paths_and_labels():
    paths, labels = [], []
    for subset in SUBSETS:
        subset_dir = os.path.join(BASE_PATH, subset)
        for fname in os.listdir(subset_dir):
            if fname.endswith(".hea"):
                path = os.path.join(subset_dir, fname[:-4])
                age, sex, codes = parse_header(path + ".hea")
                vec = np.zeros(len(CLASS2IDX), dtype=np.float32)
                for c in codes:
                    if c in CLASS2IDX:
                        vec[CLASS2IDX[c]] = 1.0
                if np.any(vec):
                    x, meta = wfdb.rdsamp(path)
                    sig = x.T.astype(np.float32)
                    fs = meta['fs'] if isinstance(meta, dict) else meta.fs
                    if fs != TARGET_FS:
                        sig = resample(sig, int(sig.shape[1] * TARGET_FS / fs), axis=1)
                    if sig.shape[1] >= SEG_LEN:
                        paths.append(path)
                        labels.append((vec, age, sex))
    return paths, labels



# === Model ===
class SEBlock(nn.Module):
    def __init__(self, c, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // reduction), nn.ReLU(), nn.Linear(c // reduction, c), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.se = SEBlock(out_c)
        self.down = nn.Sequential()
        if in_c != out_c or stride != 1:
            self.down = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm1d(out_c))

    def forward(self, x):
        identity = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) + identity
        return self.relu(out)

class SE_ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(130, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(torch.cat([x, demo], dim=1))

class ECGDatasetMultiLabelWithDemo(Dataset):
    def __init__(self, data, seg_len=SEG_LEN, target_fs=TARGET_FS):
        self.data = data
        self.seg_len = seg_len
        self.target_fs = target_fs

    def __len__(self):
        return len(self.data)

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
            return None
        return sig[:, :self.seg_len]

    def __getitem__(self, idx):
        path, (label, age, sex) = self.data[idx]
        try:
            sig, fs = self.load_record(path)
            sig = self.resample_sig(sig, fs)
            sig = self.wavelet_denoise(sig, wavelet=wavelet, start_thresh_level=start_thresh_level, threshold_scale=threshold_scale, threshold_mode=threshold_mode)
            sig = self.trunc(sig)
            if sig is None:
                return None
            sig = self.normalize(sig)
            if np.isnan(sig).any() or not np.isfinite(sig).all():
                return None

            x_tensor = torch.tensor(sig[LEAD_IDX:LEAD_IDX+1], dtype=torch.float32)
            demo_tensor = torch.tensor([age, sex], dtype=torch.float32)
            y_tensor = torch.tensor(label, dtype=torch.float32)
            return x_tensor, demo_tensor, y_tensor
        except:
            return None


# === Train ===
def train(model, train_loader, val_loader, criterion, optimizer, scheduler):


    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for x, demo, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            x, demo, y = x.to(DEVICE), demo.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, demo)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, demo, y in val_loader:
                x, demo = x.to(DEVICE), demo.to(DEVICE)
                out = model(x, demo)
                prob = torch.sigmoid(out).cpu().numpy()
                pred = (prob > 0.64 ).astype(int)
                if pred.sum() == 0:
                    pred[np.argmax(prob)] = 1
                y_true.append(y.numpy())
                y_pred.append(pred)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        f1 = f1_score(y_true, y_pred, average='macro')
        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} F1={f1:.4f} Precision={p:.4f} Recall={r:.4f}")

        # === Per-class confusion matrices ===
        cm = multilabel_confusion_matrix(y_true, y_pred)
        for i, label in enumerate(CLASS_NAMES):
            tn, fp, fn, tp = cm[i].ravel()
            print(f"{label}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        # === Histogram liczby predykcji na klasę ===
        class_counts = y_pred.sum(axis=0)
        plt.figure(figsize=(10, 4))
        plt.bar(CLASS_NAMES, class_counts)
        plt.title(f"Predicted counts per class – Epoch {epoch + 1}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"histogram_epoch_{epoch + 1}.png")
        plt.close()

        # === Confusion Matrix visualizations ===
        fig, axes = plt.subplots(nrows=2, ncols=(len(CLASS_NAMES) + 1) // 2, figsize=(16, 8))
        axes = axes.flatten()
        for i, label in enumerate(CLASS_NAMES):
            tn, fp, fn, tp = cm[i].ravel()
            matrix = np.array([[tn, fp], [fn, tp]])
            ax = axes[i]
            im = ax.imshow(matrix, cmap='Blues')
            ax.set_title(label)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Neg', 'Pos'])
            ax.set_yticklabels(['Neg', 'Pos'])
            for j in range(2):
                for k in range(2):
                    ax.text(k, j, matrix[j, k], ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(f"conf_matrices_epoch_{epoch + 1}.png")
        plt.close()

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "models/model_multi.pt")
        scheduler.step(total_loss)

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# === Main ===
if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)

    paths, labels = load_multi_label_paths_and_labels()
    data = list(zip(paths, labels))
    train_data, val_data = train_test_split(data, test_size=VAL_RATIO, random_state=SEED)

    train_labels_bin = np.array([label for _, (label, _, _) in train_data])
    pos_weight = (len(train_labels_bin) - train_labels_bin.sum(axis=0)) / (train_labels_bin.sum(axis=0) + 1e-5)

    model = SE_ResNet1D(num_classes=len(CLASS2IDX)).to(DEVICE)
    #model = SE_ResNet1D_LSTM(num_classes=len(CLASS2IDX)).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    #train_loader = DataLoader(ECGDatasetMultiLabel(train_data), batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(ECGDatasetMultiLabel(val_data), batch_size=BATCH_SIZE)

    train_loader = DataLoader(
        ECGDatasetMultiLabelWithDemo(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        ECGDatasetMultiLabelWithDemo(val_data),
        batch_size=BATCH_SIZE,
        collate_fn=collate_skip_none
    )


    train(model, train_loader, val_loader, criterion, optimizer, scheduler)
