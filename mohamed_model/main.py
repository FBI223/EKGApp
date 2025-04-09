import os
import numpy as np
import wfdb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from biosppy.signals import ecg
import seaborn as sns
from scipy.signal import resample

TARGET_FS=125
WINDOW_SIZE=188
CLASS_COUNT=10_000
BATCH_SIZE=32
EPOCHS=10

def resample_signal(signal, orig_fs, target_fs=TARGET_FS):
    if orig_fs != target_fs:
        new_len = int(len(signal) * target_fs / orig_fs)
        signal = resample(signal, new_len)
    return signal


def detect_qrs(signal, fs=TARGET_FS):
    # biosppy wymaga float i fs w Hz
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
    r_peaks = out['rpeaks']
    return r_peaks


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def augment_class(X, y, target_class, n_needed):
    class_indices = np.where(y == target_class)[0]
    new_samples = []
    while len(new_samples) < n_needed:
        i = np.random.choice(class_indices)
        signal = X[i].copy()
        noise = np.random.normal(0, 0.05, size=signal.shape)
        jitter = np.roll(signal, np.random.randint(-2, 3))
        augmented = jitter + noise
        new_samples.append(augmented)
    return np.array(new_samples), np.array([target_class] * n_needed)

def balance_classes(X, y, n_target=CLASS_COUNT):
    X_aug, y_aug = [X], [y]
    for label in np.unique(y):
        current_count = np.sum(y == label)
        if current_count < n_target:
            X_new, y_new = augment_class(X, y, label, n_target - current_count)
            X_aug.append(X_new)
            y_aug.append(y_new)
    X_balanced = np.concatenate(X_aug, axis=0)
    y_balanced = np.concatenate(y_aug, axis=0)
    return X_balanced, y_balanced




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


def extract_beats_classic(signal, r_peaks, window_size=WINDOW_SIZE):
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

        orig_fs = record.fs
        signal_raw = record.p_signal[:, 0]
        signal = resample_signal(signal_raw, orig_fs, target_fs=TARGET_FS)
        signal = preprocess_signal(signal)

        # Przeskaluj adnotacje do nowej częstotliwości
        scale = TARGET_FS / orig_fs
        ann_sample_rescaled = (ann.sample * scale).astype(int)

        # Detekcja QRS po resamplingu
        #r_peaks = detect_qrs(signal, fs=TARGET_FS)
        r_peaks = ann_sample_rescaled
        beats, beat_indices = extract_beats(signal, r_peaks, window_size=WINDOW_SIZE)

        for beat, r in zip(beats, beat_indices):
            diff = np.abs(ann_sample_rescaled - r)
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
    # Ścieżka do MIT-BIH
    mitdb_path = '../databases/mitdb'
    X, y = load_data(mitdb_path)

    # Augmentacja i balansowanie klas
    X, y = balance_classes(X, y, n_target=CLASS_COUNT)
    y = to_categorical(y, num_classes=5)
    print("Liczba uderzeń po augmentacji:", X.shape[0])

    # Dopasowanie kształtu wejścia
    X = X[..., np.newaxis]

    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=np.argmax(y, axis=1), random_state=42
    )

    # Budowa modelu
    model = build_model(input_shape=(WINDOW_SIZE, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.75),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Trenowanie modelu
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Zapis historii treningu
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    plot_history(history)

    # Ewaluacja
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy:", test_acc)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, list(label_map.keys()))
    print("Confusion Matrix:\n", cm)

    report = classification_report(y_true, y_pred, target_names=list(label_map.keys()))
    print("Classification Report:\n", report)

    # Zapis pełnego modelu końcowego
    model.save('cnn_ecg_model.keras')


if __name__ == '__main__':
    main()
