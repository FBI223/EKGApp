import wfdb
import numpy as np
from biosppy.signals import ecg
from scipy.signal import resample
import tensorflow as tf

# --- Stałe ---
TARGET_FS = 125
WINDOW_SIZE = 188
HALF_WINDOW = WINDOW_SIZE // 2
CLASS_NAMES = ['N', 'SVP', 'PVC', 'FVN', 'FPN']
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

# --- Biosppy R-peak detection ---
def detect_qrs(signal, fs):
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
    return out['rpeaks']

# --- Segment extraction z paddingiem ---
def extract_segment(signal, r, window=WINDOW_SIZE):
    beat = np.zeros(window)
    left = r - HALF_WINDOW
    right = r + HALF_WINDOW
    seg_start = max(0, left)
    seg_end = min(right, len(signal))
    seg = signal[seg_start:seg_end]
    offset = 0 if left >= 0 else -left
    beat[offset:offset+len(seg)] = seg
    return beat

# --- Wczytanie modelu ---
class ECGPredictor:
    def __init__(self, model_path='cnn_ecg_model.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES

    def predict(self, segments):
        X = np.expand_dims(segments, axis=-1)
        probs = self.model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds

# --- Porównanie modelu z adnotacjami ---
def validate_predictions(record_path, channel=0, distance_thresh=50):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, channel]
    orig_fs = record.fs

    # Resample i normalizacja
    signal = resample(signal, int(len(signal) * TARGET_FS / orig_fs)) if orig_fs != TARGET_FS else signal
    signal = preprocess_signal(signal)
    ann_sample_scaled = (ann.sample * TARGET_FS / orig_fs).astype(int)

    r_peaks = detect_qrs(signal, TARGET_FS)
    predictor = ECGPredictor()

    y_true = []
    y_pred = []

    print(f"Analyzing: {record_path}\n")

    for r in r_peaks:
        distances = np.abs(ann_sample_scaled - r)
        idx = np.argmin(distances)
        if distances[idx] > distance_thresh:
            continue

        ann_sym = ann.symbol[idx]
        if ann_sym not in AAMI_MAP:
            continue

        true_class = LABEL_MAP[AAMI_MAP[ann_sym]]
        segment = extract_segment(signal, r)
        pred_class = predictor.predict(np.array([segment]))[0]

        result = '✅' if pred_class == true_class else '❌'
        print(f"Sample {r:06d}: true={CLASS_NAMES[true_class]}, pred={CLASS_NAMES[pred_class]} {result}")

        y_true.append(true_class)
        y_pred.append(pred_class)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.mean(y_true == y_pred)

    print(f"\nTotal matched beats: {len(y_true)}")
    print(f"Accuracy: {acc * 100:.2f}%")

    total_ann = sum(1 for s in ann.symbol if s in AAMI_MAP)
    print(f"Total AAMI-annotated beats: {total_ann}")
    print(f"Matched beats (biosppy + atr): {len(y_true)}")
    print(f"Coverage: {len(y_true) / total_ann * 100:.2f}%")


if __name__ == '__main__':
    record_path = '/home/msztu223/PycharmProjects/ECG_PROJ/databases/svdb/820'
    validate_predictions(record_path)
