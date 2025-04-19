
import os
import numpy as np
import wfdb
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess




AAMI_CLASSES = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q']
}

def map_annotation_to_aami(symbol):
    for key, values in AAMI_CLASSES.items():
        if symbol in values:
            return key
    return None

def load_mitdb_segmented(path, original_fs=360, target_fs=125, segment_sec=1.2, interpolated_len=187):
    records = [f[:-4] for f in os.listdir(path) if f.endswith('.dat')]
    X, y = [], []

    for record in records:
        try:
            record_path = os.path.join(path, record)
            signal_data, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            sig = signal_data[:, 0]
            duration = len(sig) / original_fs
            num_target_samples = int(duration * target_fs)
            resampled_sig = signal.resample(sig, num_target_samples)

            scale = target_fs / original_fs
            new_peaks = [int(p * scale) for p in annotation.sample]

            segment_len = int(target_fs * segment_sec)
            half_seg = segment_len // 2

            for i, peak in enumerate(new_peaks):
                label = map_annotation_to_aami(annotation.symbol[i])
                if label is None:
                    continue

                start = peak - half_seg
                end = peak + half_seg
                if start < 0 or end > len(resampled_sig):
                    continue

                beat_segment = resampled_sig[start:end]
                interpolated = signal.resample(beat_segment, interpolated_len)
                X.append(interpolated)
                y.append(label)

        except Exception as e:
            print(f"Błąd w rekordzie {record}: {e}")

    return np.array(X), np.array(y)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    path_to_mitdb = "../databases/mitdb"
    X, y = load_mitdb_segmented(path_to_mitdb)

    classes = sorted(list(set(y)))
    class_map = {cls: i for i, cls in enumerate(classes)}
    y_numeric = np.array([class_map[label] for label in y])

    X = X.reshape(-1, 187, 1).astype(np.float32)
    y_cat = tf.keras.utils.to_categorical(y_numeric, num_classes=len(classes))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_numeric, random_state=42)

    # LSTM z implementation=1, TFLite compatible
    model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(8, kernel_size=5, activation='relu', input_shape=(187, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=2)

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Macierz pomyłek AAMI - LSTM compat")
    plt.savefig("macierz_pomylek_lstm_fixed.png")
    plt.close()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    subprocess.run(["xxd", "-i", "model.tflite", "model.cc"])


if __name__ == "__main__":
    main()
