import os
import numpy as np
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from consts import EPOCHS, BATCH_SIZE, NSRDB_PATH, MITDB_PATH, WINDOW_SIZE
import tensorflow as tf

# --- Ekstrakcja beatów z rekordu ---
def extract_beats(record_path, label_func):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]
    beats, labels = [], []
    for i in range(1, len(annotation.sample) - 1):
        sym = annotation.symbol[i]
        prev = annotation.sample[i - 1] + 20
        next_ = annotation.sample[i + 1] - 20
        if prev < next_ and next_ <= len(signal):
            segment = signal[prev:next_].astype(np.float32)
            if len(segment) != WINDOW_SIZE:
                segment = np.interp(np.linspace(0, len(segment) - 1, WINDOW_SIZE),
                                    np.arange(len(segment)), segment)
            beats.append(segment)
            labels.append(label_func(sym))
    return beats, labels

# --- Funkcja etykietująca ---
def binary_label(symbol):
    return 1 if symbol == 'N' else 0

# --- Zbuduj zbiór danych ---
def build_binary_dataset():
    X, y = [], []
    for db_path in [NSRDB_PATH, MITDB_PATH]:
        records = [f[:-4] for f in os.listdir(db_path) if f.endswith('.dat')]
        for rec in tqdm(records, desc=f"Loading {os.path.basename(db_path)}"):
            path = os.path.join(db_path, rec)
            try:
                beats, labels = extract_beats(path, binary_label)
                X.extend(beats)
                y.extend(labels)
            except:
                continue
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y

# --- Budowa modelu CNN ---
def build_model():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        Conv1D(32, 5, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='elu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='elu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Wizualizacja macierzy pomyłek ---
def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



def train_model_binary():

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(gpus)

    # --- Główny pipeline ---
    X, y = build_binary_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)


    # --- Save model ---
    model.save("binary_normal_abnormal_model.keras")
    print("✅ Model saved as binary_normal_abnormal_model.keras")


    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, preds))
    plot_conf_matrix(y_test, preds)



if __name__ == "__main__":
    train_model_binary()

