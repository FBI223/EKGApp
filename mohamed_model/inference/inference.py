import wfdb
import numpy as np
from biosppy.signals import ecg
from scipy.signal import resample
import tensorflow as tf

# --- Stałe ---
TARGET_FS = 125
WINDOW_SIZE = 188
CLASS_NAMES = ['N', 'SVP', 'PVC', 'FVN', 'FPN']
MODEL_PATH = 'cnn_ecg_model.keras'
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'SVP', 'a': 'SVP', 'J': 'SVP', 'S': 'SVP',
    'V': 'PVC', 'E': 'PVC',
    'F': 'FVN',
    '/': 'FPN'
}
LABEL_MAP = {'N': 0, 'SVP': 1, 'PVC': 2, 'FVN': 3, 'FPN': 4}

# --- Normalizacja ---
def preprocess_signal(signal):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal

# --- Detekcja QRS ---
def detect_qrs(signal, fs):
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
    return out['rpeaks']

# --- Ekstrakcja beatów ---
def extract_beats(signal, r_peaks, window_size=WINDOW_SIZE):
    half_window = window_size // 2
    beats, indices = [], []
    for r in r_peaks:
        left = r - half_window
        right = r + half_window
        beat = np.zeros(window_size)
        seg_start = max(0, left)
        seg_end = min(right, len(signal))
        seg = signal[seg_start:seg_end]
        offset = 0 if left >= 0 else -left
        beat[offset:offset+len(seg)] = seg
        beats.append(beat)
        indices.append(r)
    return np.array(beats), np.array(indices)

# --- Klasa do predykcji ---
class ECGPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES

    def predict(self, segments):
        X = segments[..., np.newaxis]  # (n, 188, 1)
        probs = self.model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds, probs

# --- Główna funkcja walidacji ---
def evaluate_predictions(record_path, channel=0, distance_thresh=50):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, channel]
    orig_fs = record.fs
    scale = TARGET_FS / orig_fs

    signal = resample(signal, int(len(signal) * scale)) if orig_fs != TARGET_FS else signal
    signal = preprocess_signal(signal)
    ann_scaled = (ann.sample * scale).astype(int)

    r_peaks = detect_qrs(signal, TARGET_FS)
    beats, beat_locs = extract_beats(signal, r_peaks)

    predictor = ECGPredictor()
    preds, _ = predictor.predict(beats)

    correct = total = 0

    for r, pred_class in zip(beat_locs, preds):
        distances = np.abs(ann_scaled - r)
        idx = np.argmin(distances)
        if distances[idx] > distance_thresh:
            continue
        sym = ann.symbol[idx]
        true_aami = AAMI_MAP.get(sym)
        if true_aami is None:
            continue
        true_class = LABEL_MAP[true_aami]
        total += 1
        result = '✅' if pred_class == true_class else '❌'
        if pred_class == true_class:
            correct += 1
        print(f"Sample {r:06d}: true={CLASS_NAMES[true_class]}, pred={CLASS_NAMES[pred_class]} {result}")

    if total:
        print(f"\nMatched beats: {total}")
        print(f"Accuracy: {100 * correct / total:.2f}%")
    else:
        print("No matching annotations found.")

# --- Uruchomienie ---
if __name__ == '__main__':
    evaluate_predictions('/home/msztu223/PycharmProjects/ECG_PROJ/databases/svdb/821')