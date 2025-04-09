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


def display_annotations(file_prefix, channel=0):
    record = wfdb.rdrecord(file_prefix)
    ann = wfdb.rdann(file_prefix, 'atr')

    signal = record.p_signal[:, channel]
    orig_fs = record.fs
    scale = TARGET_FS / orig_fs

    # Resampling sygnału (dla porównania)
    resampled_signal = resample(signal, int(len(signal) * scale)) if orig_fs != TARGET_FS else signal

    # Przeskalowane adnotacje
    ann_sample_scaled = (ann.sample * scale).astype(int)

    print(f"Plik: {file_prefix} | Próbkowanie: {orig_fs} Hz → {TARGET_FS} Hz")
    print(f"Liczba adnotacji: {len(ann.sample)}\n")

    for i, (orig_s, scaled_s, sym) in enumerate(zip(ann.sample, ann_sample_scaled, ann.symbol)):
        aami = AAMI_MAP.get(sym, '?')
        print(f"Beat {i + 1:04d}: orig={orig_s:06d}, resampled={scaled_s:06d}, sym='{sym}', AAMI='{aami}'")


# --- Normalizacja jak w treningu ---
def preprocess_signal(signal):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal

# --- Detekcja QRS za pomocą biosppy ---
def detect_qrs(signal, fs):
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
    return out['rpeaks']

# --- Ekstrakcja beatów z paddingiem ---
def extract_beats(signal, r_peaks, window_size=WINDOW_SIZE):
    half_window = window_size // 2
    beats = []
    indices = []
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

# --- Przetwarzanie pliku .dat/.hea ---
def load_and_process_record(file_path, channel=0):
    record = wfdb.rdrecord(file_path)
    signal = record.p_signal[:, channel]
    fs = record.fs
    if fs != TARGET_FS:
        signal = resample(signal, int(len(signal) * TARGET_FS / fs))
    signal = preprocess_signal(signal)
    r_peaks = detect_qrs(signal, fs=TARGET_FS)
    beats, beat_locs = extract_beats(signal, r_peaks)
    return beats, beat_locs

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

# --- Główna funkcja inference ---
def run_prediction(file_prefix):
    beats, beat_locs = load_and_process_record(file_prefix)
    predictor = ECGPredictor()
    preds, probs = predictor.predict(beats)

    for i, (r, pred_id) in enumerate(zip(beat_locs, preds)):
        label = predictor.class_names[pred_id]
        print(f"Beat {i+1:04d} at sample {r:06d}: predicted → {label}")



if __name__ == '__main__':
    run_prediction('820')  # używa plików 100.hea, 100.dat, 100.atr
    display_annotations('820')  # 100.hea, 100.dat, 100.atr muszą być w katalogu

