import wfdb
import numpy as np
from tensorflow.keras.models import load_model
from consts import WINDOW_SIZE


ANNOTATION_MAP = {
    'N': 0, 'V': 1, '/': 2, 'R': 3, 'L': 4, 'S': 5, '!': 6, 'E': 7
}
INV_ANNOTATION_MAP = {v: k for k, v in ANNOTATION_MAP.items()}


def extract_beats(record_path):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]  # use only channel 0
    beats, labels = [], []
    for i in range(1, len(annotation.sample) - 1):
        sym = annotation.symbol[i]
        if sym in ANNOTATION_MAP:
            prev = annotation.sample[i - 1] + 20
            next_ = annotation.sample[i + 1] - 20
            if prev < next_ and next_ <= len(signal):
                segment = signal[prev:next_].astype(np.float32)
                if len(segment) != WINDOW_SIZE:
                    segment = np.interp(np.linspace(0, len(segment) - 1, WINDOW_SIZE),
                                        np.arange(len(segment)), segment)
                beats.append(segment)
                labels.append(ANNOTATION_MAP[sym])
    return beats, labels

# --- Pipeline wykrycia beatów i predykcji ---
def process_ecg_file(record_path,model):
    segments , labels = extract_beats(record_path)
    predictions = []

    for i in range(1, len(segments) - 1):
        beat = segments[i].reshape(1, WINDOW_SIZE, 1).astype(np.float32)
        pred = model.predict(beat, verbose=0)
        pred_class = np.argmax(pred)
        is_ok_prediction = pred_class == labels[i]
        predictions.append((i, pred_class, INV_ANNOTATION_MAP[pred_class], labels[i] , INV_ANNOTATION_MAP[labels[i]] , is_ok_prediction ))

    return predictions  # lista (qrs_index, klasa numeryczna, symbol, realna klasa)





def predict(record_path,model_path):
    model = load_model(model_path)
    predictions = process_ecg_file(record_path,model)
    for i in range(len(predictions)):
        print(predictions[i])

