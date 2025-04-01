import os
import numpy as np
import wfdb
import json
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

# Mapping from MIT-BIH annotation symbols to AAMI classes and label encoding
aami_mapping = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'SVP', 'a': 'SVP', 'J': 'SVP', 'S': 'SVP',
    'V': 'PVC', 'E': 'PVC',
    'F': 'FVN',
    '/': 'FPN'
}
label_map = {'N': 0, 'SVP': 1, 'PVC': 2, 'FVN': 3, 'FPN': 4}


def preprocess_signal(signal):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def detect_qrs(signal, fs=125):
    # Minimum distance ~200ms
    distance = int(0.2 * fs)
    # Use threshold relative to max value
    peaks, _ = find_peaks(signal, distance=distance, height=0.5 * np.max(signal))
    return peaks


def extract_beats(signal, r_peaks, window_size=188):
    half_window = window_size // 2
    beats = []
    indices = []
    for r in r_peaks:
        if r - half_window >= 0 and r + half_window < len(signal):
            beat = signal[r - half_window: r + half_window]
            beats.append(beat)
            indices.append(r)
    return np.array(beats), np.array(indices)


def load_data(mitdb_path):
    X, y = [], []
    files = [f[:-4] for f in os.listdir(mitdb_path) if f.endswith('.hea')]
    for rec in files:
        record_path = os.path.join(mitdb_path, rec)
        try:
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print("Error reading record", rec, e)
            continue
        # Use first channel and preprocess
        signal = preprocess_signal(record.p_signal[:, 0])
        r_peaks = detect_qrs(signal, fs=record.fs)
        beats, beat_indices = extract_beats(signal, r_peaks, window_size=188)
        # Associate each beat with the closest annotation
        for beat, r in zip(beats, beat_indices):
            diff = np.abs(ann.sample - r)
            idx = np.argmin(diff)
            ann_symbol = ann.symbol[idx]
            if ann_symbol in aami_mapping:
                mapped_label = aami_mapping[ann_symbol]
                X.append(beat)
                y.append(label_map[mapped_label])
    X = np.array(X)
    y = np.array(y)
    return X, y


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=5, strides=1, activation='relu',
                               padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=5, strides=1, padding='same'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=5, strides=1, padding='same'),
        tf.keras.layers.Conv1D(128, kernel_size=5, strides=1, activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=5, strides=1, padding='same'),
        tf.keras.layers.Conv1D(256, kernel_size=5, strides=1, activation='relu',
                               padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model


def plot_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.savefig('training_accuracy.png')
    plt.close()


def main():
    # Załóż, że MITDB znajduje się w ../databases/mitdb
    mitdb_path = '../databases/mitdb'
    X, y = load_data(mitdb_path)
    print("Liczba uderzeń:", X.shape[0])
    # Dopasuj wymiar: (num_samples, 188, 1)
    X = X[..., np.newaxis]

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    # Budowa modelu CNN
    model = build_model(input_shape=(188, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Trenowanie modelu
    history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Zapis statystyk treningu
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    plot_history(history)

    # Ewaluacja na zbiorze testowym
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy:", test_acc)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    report = classification_report(y_test, y_pred, target_names=list(label_map.keys()))
    print("Classification Report:\n", report)

    # Zapis modelu
    model.save('cnn_ecg_model.h5')


if __name__ == '__main__':
    main()
