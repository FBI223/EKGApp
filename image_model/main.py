import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import tensorflow as tf
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm  # upewnij się, że masz ten import

# --- CONSTS ---
MITDB_PATH = '../databases/mitdb'
WINDOW_SIZE = 128
FOLDS = 10
EPOCHS = 30
BATCH_SIZE = 128
ANNOTATION_MAP = {
    'N': 0,  # Normal
    'L': 1,  # Left bundle branch block beat
    'R': 2,  # Right bundle branch block beat
    'A': 3,  # Atrial premature beat (PAC)
    'V': 4,  # Premature ventricular contraction (PVC)
    '/': 5,  # Paced beat
}
INV_ANNOTATION_MAP = {i: symbol for symbol, i in ANNOTATION_MAP.items()}


# --- Extract beats and convert to 2D images ---
def extract_beat_images(record_path):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]
    beats, labels = [], []

    for i in range(1, len(ann.sample) - 1):
        sym = ann.symbol[i]
        if sym not in ANNOTATION_MAP:
            continue
        prev = ann.sample[i - 1] + 20
        next_ = ann.sample[i + 1] - 20
        if prev < next_ and next_ <= len(signal):
            segment = signal[prev:next_]
            segment = np.interp(np.linspace(0, len(segment)-1, WINDOW_SIZE),
                                np.arange(len(segment)), segment)

            fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
            canvas = FigureCanvas(fig)
            plt.plot(segment, color='black')
            plt.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            canvas.draw()

            image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape((128, 128, 4))[:, :, :3]  # RGB bez kanału alfa
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # na 1 kanał

            plt.close(fig)

            beats.append(image)
            labels.append(ANNOTATION_MAP[sym])  # <-- TO BYŁO POMINIĘTE

    return beats, labels




def build_dataset():
    os.makedirs("preprocessed_data", exist_ok=True)

    x_path = "preprocessed_data/X.npy"
    y_path = "preprocessed_data/y.npy"
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("✅ Wczytywanie zapisanych danych...")
        X = np.load(x_path)
        y = np.load(y_path)
        return X, y

    X, y = [], []
    records = [f[:-4] for f in os.listdir(MITDB_PATH) if f.endswith(".dat")]
    print(f"Znalezione rekordy: {records}")

    for rec in tqdm(records, desc="⏳ Przetwarzanie rekordów"):
        try:
            path = os.path.join(MITDB_PATH, rec)
            beats, labels = extract_beat_images(path)
            X.extend(beats)
            y.extend(labels)
        except Exception as e:
            print(f"Błąd rekordu {rec}: {e}")

    X = np.array(X, dtype=np.uint8)[..., np.newaxis] / 255.0
    y = np.array(y)
    print(f"📦 Załadowano: {len(X)} beatów")

    np.save(x_path, X)
    np.save(y_path, y)
    print("💾 Dane zapisane do preprocessed_data/")

    return X, y




# --- Build CNN model ---
def build_model():
    model = Sequential([
        Input(shape=(128, 128, 1)),
        Conv2D(64, kernel_size=3, padding='same', activation='elu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, padding='same', activation='elu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),

        Conv2D(128, kernel_size=3, padding='same', activation='elu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=3, padding='same', activation='elu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),

        Conv2D(256, kernel_size=3, padding='same', activation='elu'),
        BatchNormalization(),
        Conv2D(256, kernel_size=3, padding='same', activation='elu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),

        Flatten(),
        Dense(2048, activation='elu'),
        Dropout(0.5),
        Dense(len(ANNOTATION_MAP), activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Confusion matrix ---
def plot_conf_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))],
                      columns=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))])
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({name})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_{name}.png")
    plt.close()


# --- Train model ---
def train_model():
    X, y = build_dataset()
    if len(X) == 0:
        print("❌ Brak danych. Sprawdź ścieżkę MITDB_PATH lub zawartość folderu.")
        return

    y_cat = to_categorical(y, num_classes=len(ANNOTATION_MAP))
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        model = build_model()
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_cat[train_idx], y_cat[val_idx]

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ModelCheckpoint(f"model_fold_{fold}.keras", save_best_only=True)
        ]

        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

        y_pred = np.argmax(model.predict(X_val), axis=1)
        y_true = np.argmax(y_val, axis=1)

        print(classification_report(y_true, y_pred, target_names=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))]))
        plot_conf_matrix(y_true, y_pred, f"fold_{fold}")

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    print("\n=== GLOBAL EVALUATION ===")
    print(classification_report(all_y_true, all_y_pred, target_names=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))]))
    plot_conf_matrix(all_y_true, all_y_pred, "global")


# --- Run ---
if __name__ == "__main__":
    train_model()