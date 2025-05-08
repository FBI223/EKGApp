# ECG Arrhythmia 1D‑CNN Pipeline (PyTorch)
# MIT‑BIH Arrhythmia Database – 10‑second fragment classification (17 classes)
# Based on: Yıldırım et al., "Arrhythmia detection using deep CNN with long‑duration ECG signals" (Computers in Biology and Medicine, 2018)
# Annotation codes: https://archive.physionet.org/physiobank/database/html/mitdbdir/tables.htm
# Author: ChatGPT – updated 2025‑05‑07

"""
Usage:
    python ecg_1dcnn_pipeline.py --data ../databases/mitdb --epochs 50 --batch 32
"""

import argparse
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import wfdb
from tqdm import tqdm
from collections import Counter

SAMPLE_RATE = 360
SEG_LEN = SAMPLE_RATE * 10
NUM_CLASSES = 17






SYM2CLS = {
    "N": 0,  "A": 1,  "J": 1,  "F": 2,  "AFIB": 3,  "SVTA": 4,  "/": 5,  "V": 6,
    "B": 7,  "T": 8,  "VT": 9,  "IVR": 10, "VF": 11, "!": 12, "L": 13, "R": 14,
    "E": 15, "/p": 16,
}

RHYTHM_MAP = {
    "(AFIB": 3, "(SVTA": 4, "(B": 7, "(T": 8, "(VT": 9,
    "(IVR": 10, "(VFL": 11, "(BII": 15, "(P": 16
}

def segment_record(hea_path: Path):
    name = hea_path.with_suffix("")
    rec = wfdb.rdrecord(str(name))
    if "MLII" not in rec.sig_name:
        return []
    ml_idx = rec.sig_name.index("MLII")
    sig = rec.p_signal[:, ml_idx]
    sig = MinMaxScaler((-1, 1)).fit_transform(sig.reshape(-1, 1)).flatten()
    ann = wfdb.rdann(str(name), "atr")

    out = []
    for i in range(len(sig) // SEG_LEN):
        s, e = i * SEG_LEN, (i + 1) * SEG_LEN
        frag = sig[s:e]
        if frag.size < SEG_LEN:
            continue

        beat_syms = np.array(ann.symbol)[(ann.sample >= s) & (ann.sample < e)]
        beat_filtered = [b for b in beat_syms if b in SYM2CLS]

        aux_syms = np.array(ann.aux_note)[(ann.sample >= s) & (ann.sample < e)]
        rhythm_filtered = [r for r in aux_syms if r in RHYTHM_MAP]

        # Priorytet: rytm > beat
        if rhythm_filtered:
            rhythm = Counter(rhythm_filtered).most_common(1)[0][0]
            out.append((frag.astype(np.float32), RHYTHM_MAP[rhythm]))
        elif beat_filtered:
            beat = Counter(beat_filtered).most_common(1)[0][0]
            out.append((frag.astype(np.float32), SYM2CLS[beat]))

    return out


class MITBIHFragmentDataset(Dataset):
    def __init__(self, root: Path, split=(0.7, 0.15, 0.15), mode="train"):
        assert mode in {"train", "val", "test"}
        cache = root / "fragments.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            print(f"[DEBUG] x dtype: {data['x'].dtype}, shape: {data['x'].shape}")

            self.samples = list(zip(data["x"], data["y"]))
        else:
            frags = []
            label_counts = {v: 0 for v in SYM2CLS.values()}
            for hea in tqdm(list(root.glob("*.hea")), desc="Segmenting records"):
                segs = segment_record(hea)
                for frag, label in segs:
                    label_counts[label] += 1
                frags.extend(segs)

            if not frags:
                raise RuntimeError(
                    "Brak poprawnych fragmentów MLII – brak danych lub nieobsługiwane symbole adnotacji.")

            print("\n[DEBUG] Podsumowanie fragmentów na klasy:")
            for cls_id, count in sorted(label_counts.items()):
                print(f"    Klasa {cls_id:2d}: {count:4d} segmentów")

            # Wykres
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            labels = sorted(label_counts.keys())
            counts = [label_counts[k] for k in labels]
            plt.bar(labels, counts)
            plt.xlabel("Klasa")
            plt.ylabel("Liczba fragmentów")
            plt.title("Rozkład fragmentów ECG na klasy")
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("class_distribution.png")
            print("Wykres zapisany → class_distribution.png")

            # CSV
            with open("class_distribution.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["class_id", "count"])
                for cls_id in labels:
                    writer.writerow([cls_id, label_counts[cls_id]])
            print("CSV zapisany → class_distribution.csv")

            xs, ys = zip(*frags)
            xs = np.array(xs, dtype=np.float32)  # wymuszenie typu
            ys = np.array(ys, dtype=np.int64)
            np.savez_compressed(cache, x=xs, y=ys)

            self.samples = list(zip(xs, ys))

        rng = np.random.default_rng(42)
        rng.shuffle(self.samples)
        n = len(self.samples)
        n_train, n_val = int(split[0] * n), int(split[1] * n)
        if mode == "train":
            self.samples = self.samples[:n_train]
        elif mode == "val":
            self.samples = self.samples[n_train:n_train + n_val]
        else:
            self.samples = self.samples[n_train + n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x).unsqueeze(0), y

class ECG1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 128, 50, 3), nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2, 3),
            nn.Conv1d(128, 32, 7), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 32, 10), nn.ReLU(),
            nn.Conv1d(32, 128, 5, 2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(128, 256, 15), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(256, 512, 5), nn.ReLU(),
            nn.Conv1d(512, 128, 3), nn.ReLU(),
        )
        with torch.no_grad():
            feat_len = self.features(torch.zeros(1, 1, SEG_LEN)).shape[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_len, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def run_epoch(model, loader, crit, opt=None, device="cpu"):
    train = opt is not None
    model.train() if train else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def evaluate(model, loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            y_true.extend(y.numpy()); y_pred.extend(preds.numpy())
    print(classification_report(y_true, y_pred, digits=4))
    np.save("confusion_matrix.npy", confusion_matrix(y_true, y_pred))
    print("Confusion matrix saved → confusion_matrix.npy")

def save_metrics_csv(epoch_stats, path="training_metrics.csv"):
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(epoch_stats)
    print(f"Saved metrics to {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../databases/mitdb")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise FileNotFoundError(root)

    train_ds = MITBIHFragmentDataset(root, mode="train")
    val_ds = MITBIHFragmentDataset(root, mode="val")
    test_ds = MITBIHFragmentDataset(root, mode="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ECG1DCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    epoch_stats = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, crit, opt, device)
        val_loss, val_acc = run_epoch(model, val_loader, crit, None, device)
        epoch_stats.append((epoch, tr_loss, tr_acc, val_loss, val_acc))
        print(f"Epoch {epoch:03d} | TrainAcc {tr_acc:.4f} | ValAcc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "ecg1dcnn_best.pth")

    save_metrics_csv(epoch_stats)
    model.load_state_dict(torch.load("ecg1dcnn_best.pth"))
    evaluate(model, test_loader, device)
    torch.save(model.state_dict(), "ecg1dcnn_final.pth")
    print("Model weights saved → ecg1dcnn_final.pth")

if __name__ == "__main__":
    main()
