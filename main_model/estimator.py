import wfdb
import numpy as np
from scipy.signal import resample
from scipy.interpolate import interp1d
from ecgdetectors import Detectors
from tensorflow.keras.models import load_model

# --- Ustawienia ---
WINDOW_SIZE = 360  # np. 1s sygnału po interpolacji
ANNOTATION_MAP = { 'N': 0, 'V': 1, '/': 2, 'R': 3, 'L': 4, 'A': 5, '!': 6, 'E': 7 }
INV_ANNOTATION_MAP = {v: k for k, v in ANNOTATION_MAP.items()}

# --- Model ---
model = load_model("trained_models/v2/model_fold_1.keras")

# --- Preprocessing pojedynczego segmentu ---
def preprocess_segment(segment):
    if len(segment) != WINDOW_SIZE:
        f = interp1d(np.linspace(0, 1, len(segment)), segment, kind='linear')
        segment = f(np.linspace(0, 1, WINDOW_SIZE))
    return segment.reshape(1, WINDOW_SIZE, 1).astype(np.float32)

# --- Pipeline wykrycia beatów i predykcji ---
def process_ecg_file(record_path, model_sample_rate=360):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]  # Lead II
    original_fs = record.fs

    # Resampling jeśli potrzeba
    if original_fs != model_sample_rate:
        num_samples = int(len(signal) * model_sample_rate / original_fs)
        signal = resample(signal, num_samples)
        print(f"[INFO] Resampled from {original_fs} Hz to {model_sample_rate} Hz")

    # QRS detection – Pan Tompkins
    detectors = Detectors(model_sample_rate)
    qrs_peaks = detectors.pan_tompkins_detector(signal)

    predictions = []

    for i in range(1, len(qrs_peaks) - 1):  # pomijamy pierwszy i ostatni QRS
        left = qrs_peaks[i - 1] + 20
        right = qrs_peaks[i + 1] - 20
        if right > left and right <= len(signal):
            segment = signal[left:right]
            beat = preprocess_segment(segment)
            pred = model.predict(beat, verbose=0)
            pred_class = np.argmax(pred)
            predictions.append((i, pred_class, INV_ANNOTATION_MAP[pred_class]))

    return predictions  # lista (qrs_index, klasa numeryczna, symbol)

