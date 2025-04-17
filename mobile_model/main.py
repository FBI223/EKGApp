# ================================
# CNN do klasyfikacji arytmii (5 klas AAMI)
# Wejście: 188 próbek (Lead II, MIT-BIH, 360 Hz)
# Kompatybilny z TensorFlow Lite (mobilne aplikacje)
# ================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import wfdb
import os
from scipy.signal import butter,sosfiltfilt

# --- Parametry ---
DATABASE_PATH = '../databases/mitdb'
SEGMENT_LENGTH = 188
FS = 360  # Hz

# --- Butterworth filter ---


def butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=360, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)


# --- Mapowanie klas MIT-BIH → AAMI ---
aami_map = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}


def global_zscore_normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / (std + 1e-6)


# --- Załaduj i przetwórz dane z MIT-BIH ---
def load_mitdb():
    X = []
    y = []
    records = [f[:-4] for f in os.listdir(DATABASE_PATH) if f.endswith('.dat')]
    for rec in records:
        record = wfdb.rdrecord(f"{DATABASE_PATH}/{rec}", channels=[0])
        annotation = wfdb.rdann(f"{DATABASE_PATH}/{rec}", 'atr')
        signal = butter_bandpass_filter(record.p_signal.flatten())

        for i, (pos, sym) in enumerate(zip(annotation.sample, annotation.symbol)):
            if sym not in aami_map:
                continue
            if pos < SEGMENT_LENGTH // 2 or pos + SEGMENT_LENGTH // 2 > len(signal):
                continue
            segment = signal[pos - SEGMENT_LENGTH//2 : pos + SEGMENT_LENGTH//2]
            if len(segment) == SEGMENT_LENGTH:
                X.append(segment)
                y.append(aami_map[sym])

    return np.array(X), np.array(y)

X, y = load_mitdb()
X = global_zscore_normalize(X)
np.save('X_188.npy', X)
np.save('y_aami.npy', y)

X = X.reshape((-1, SEGMENT_LENGTH, 1)).astype(np.float32)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes=5)
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, stratify=y_encoded)

# --- Wykres klas ---
unique, counts = np.unique(y, return_counts=True)
plt.figure()
plt.bar(unique, counts)
plt.title('Rozkład klas AAMI')
plt.xlabel('Klasa')
plt.ylabel('Liczba próbek')
plt.savefig('class_distribution.png')

# --- Model ---
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(188, 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.6),
    Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Trening ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=128,
                    callbacks=[ModelCheckpoint('best_model.h5', save_best_only=True), early_stop])

# --- Wykres strat ---
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Strata treningowa i walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.savefig('loss_plot.png')

# --- Macierz pomyłek i raport ---
model = tf.keras.models.load_model('best_model.h5')
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure()
ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(cmap='Blues')
plt.title('Macierz pomyłek')
plt.savefig('confusion_matrix.png')

report = classification_report(y_true_labels, y_pred_labels, target_names=encoder.classes_)
print(report)

# --- Konwersja do TFLite (bez kwantyzacji) ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("ecg_cnn_model_fp32.tflite", "wb") as f:
    f.write(tflite_model)

# --- Konwersja do TFLite (z kwantyzacją 8-bit) ---
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter.convert()
with open("ecg_cnn_model_quant.tflite", "wb") as f:
    f.write(tflite_quant)
