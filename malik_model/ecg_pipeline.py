import wfdb
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from model import build_selfonn_model

AAMI_CLASSES = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q']
}

def map_aami(symbol):
    for cls, group in AAMI_CLASSES.items():
        if symbol in group:
            return cls
    return None

def extract_beats_trio(record_name, segment_len=128, fs_target=125):
    rec = wfdb.rdrecord(record_name)
    ann = wfdb.rdann(record_name, 'atr')
    signal = rec.p_signal[:, 0]
    fs = rec.fs

    # Resample
    signal_resampled = resample(signal, int(len(signal) * fs_target / fs))
    ratio = fs_target / fs
    ann_samples_resampled = [int(s * ratio) for s in ann.sample]

    beats = []
    labels = []

    for i in range(1, len(ann_samples_resampled) - 1):
        idx = ann_samples_resampled[i]
        sym = ann.symbol[i]
        label = map_aami(sym)
        if label is None: continue

        def get_seg(center):
            left = center - segment_len // 2
            right = center + segment_len // 2
            if left < 0 or right >= len(signal_resampled):
                return None
            seg = signal_resampled[left:right]
            return (seg - np.mean(seg)) / np.std(seg)

        seg0 = get_seg(ann_samples_resampled[i - 1])
        seg1 = get_seg(idx)
        seg2 = get_seg(ann_samples_resampled[i + 1])
        if seg0 is None or seg1 is None or seg2 is None:
            continue

        trio = np.stack([seg1, (seg0 + seg2) / 2], axis=1)  # (128, 2)
        beats.append(trio)
        labels.append(label)

    return np.array(beats), np.array(labels)

def load_all_beats(path='../databases/mitdb'):
    X, y = [], []
    records = sorted(glob(f'{path}/*.dat'))
    for rec_path in records:
        rec_name = os.path.basename(rec_path).split('.')[0]
        beats, labels = extract_beats_trio(f"{path}/{rec_name}")
        X.append(beats)
        y.append(labels)
    return np.concatenate(X), np.concatenate(y)

def save_if_not_exists():
    if not (os.path.exists("X.npy") and os.path.exists("y.npy")):
        X, y = load_all_beats()
        np.save("X.npy", X)
        np.save("y.npy", y)
        print("[INFO] Saved X.npy and y.npy")
    else:
        print("[INFO] X.npy and y.npy already exist")

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

def train_and_export():
    X = np.load("X.npy")
    y = np.load("y.npy")

    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

    model = build_selfonn_model(input_shape=(128, 2), num_classes=5, Q=7)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128)

    model.save("ecg_selfonn_model.h5")

    y_pred = model.predict(X_test)
    y_pred_labels = lb.inverse_transform(y_pred)
    y_true_labels = lb.inverse_transform(y_test)

    plot_confusion(y_true_labels, y_pred_labels, lb.classes_)
    print(classification_report(y_true_labels, y_pred_labels, target_names=lb.classes_))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("ecg_selfonn_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("[INFO] Exported to ecg_selfonn_model.tflite")


if __name__ == "__main__":
    save_if_not_exists()
    train_and_export()

