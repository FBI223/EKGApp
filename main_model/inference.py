import wfdb
import numpy as np
from biosppy.signals import ecg
from scipy.signal import resample
import tensorflow as tf

from main_model.consts import ANNOTATION_MAP

# --- Stałe ---
FS_TARGET = 360
WINDOW_SIZE = FS_TARGET  # jeden cykl ~ 1s
HALF_WINDOW = WINDOW_SIZE // 2
CLASS_NAMES = ['N', 'S', 'V', 'Q']
MODEL_PATH = 'model_final.h5'


# --- Normalizacja ---
def preprocess_signal(sig):
    sig = sig - np.mean(sig)
    sig = sig / np.max(np.abs(sig))
    return sig

# --- Detekcja QRS ---
def detect_qrs(signal, fs):
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
    return out['rpeaks']

# --- Segmentacja z paddingiem ---
def extract_beats(signal, r_peaks):
    beats = []
    indices = []
    for r in r_peaks:
        left = r - HALF_WINDOW
        right = r + HALF_WINDOW
        beat = np.zeros(WINDOW_SIZE)
        start = max(0, left)
        end = min(right, len(signal))
        seg = signal[start:end]
        offset = 0 if left >= 0 else -left
        beat[offset:offset+len(seg)] = seg
        beats.append(beat)
        indices.append(r)
    return np.array(beats), np.array(indices)

# --- Przetwarzanie rekordu .dat/.hea ---
def process_record(file_path, channel=0):
    record = wfdb.rdrecord(file_path)
    signal = record.p_signal[:, channel]
    orig_fs = record.fs
    if orig_fs != FS_TARGET:
        signal = resample(signal, int(len(signal) * FS_TARGET / orig_fs))
    signal = preprocess_signal(signal)
    r_peaks = detect_qrs(signal, FS_TARGET)
    beats, beat_locs = extract_beats(signal, r_peaks)
    return beats, beat_locs

# --- Klasa predykcyjna ---
class ECGPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES

    def predict(self, segments):
        X = segments[..., np.newaxis]
        probs = self.model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds, probs

# --- Główna funkcja ---
def run_prediction(record_path):
    beats, locs = process_record(record_path)
    predictor = ECGPredictor()
    preds, _ = predictor.predict(beats)

    for i, (r, cls_id) in enumerate(zip(locs, preds)):
        label = predictor.class_names[cls_id]
        print(f"Beat {i+1:04d} at sample {r:06d}: predicted → {label}")

# --- Adnotacje z pliku atr ---
def display_annotations(file_path):
    record = wfdb.rdrecord(file_path)
    ann = wfdb.rdann(file_path, 'atr')
    fs = record.fs
    scale = FS_TARGET / fs

    ann_scaled = (ann.sample * scale).astype(int)

    print(f"\nAnnotations for {file_path}:")
    for i, (s_old, s_new, sym) in enumerate(zip(ann.sample, ann_scaled, ann.symbol)):
        aami = ANNOTATION_MAP.get(sym, '?')
        print(f"Beat {i+1:04d}: orig={s_old:06d}, scaled={s_new:06d}, sym='{sym}', AAMI={aami}")

def evaluate_predictions(record_path, distance_thresh=50):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]
    orig_fs = record.fs

    # Resample and normalize
    scale = FS_TARGET / orig_fs
    signal = resample(signal, int(len(signal) * scale)) if orig_fs != FS_TARGET else signal
    signal = preprocess_signal(signal)
    ann_scaled = (ann.sample * scale).astype(int)

    # Biosppy QRS + extract beats
    r_peaks = detect_qrs(signal, FS_TARGET)
    beats, beat_locs = extract_beats(signal, r_peaks)

    predictor = ECGPredictor()
    preds, _ = predictor.predict(beats)

    correct = 0
    total = 0

    print(f"\n✅ Evaluating {record_path} predictions vs annotations:\n")

    for r, pred_class in zip(beat_locs, preds):
        distances = np.abs(ann_scaled - r)
        idx = np.argmin(distances)
        if distances[idx] > distance_thresh:
            continue  # brak adnotacji blisko

        sym = ann.symbol[idx]
        true_class = ANNOTATION_MAP.get(sym)
        if true_class is None:
            continue  # nieznana klasa

        total += 1
        pred_label = CLASS_NAMES[pred_class]
        true_label = CLASS_NAMES[true_class]

        result = '✅' if pred_class == true_class else '❌'
        if pred_class == true_class:
            correct += 1

        print(f"Sample {r:06d}: true={true_label}, pred={pred_label} {result}")

    if total > 0:
        print(f"\nMatched beats: {total}")
        print(f"Accuracy: {100 * correct / total:.2f}%")
    else:
        print("No matching annotations found within threshold.")

# --- Run main evaluation ---
if __name__ == '__main__':
    RECORD = '/home/msztu223/PycharmProjects/ECG_PROJ/databases/svdb/822'
    evaluate_predictions(RECORD)
