import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import tensorflow as tf


# --- Config ---
MITDB_PATH = '../../databases/mitdb'
WINDOW_SIZE = 360
FOLDS = 10
EPOCHS = 16
BATCH_SIZE = 32

# --- Annotation map ---
ANNOTATION_MAP = {
    'N': 0, 'V': 1, '/': 2, 'R': 3, 'L': 4, 'A': 5, '!': 6, 'E': 7
}
INV_ANNOTATION_MAP = {v: k for k, v in ANNOTATION_MAP.items()}

# --- Utility functions ---
def plot_conf_matrix(y_true, y_pred, fold):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))],
                         columns=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))])
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'conf_matrix_fold_{fold}.png')
    plt.close()

# --- Augmentation ---
def augment_1d(signal, label):
    aug_signals = [signal]
    noise = np.random.normal(0, 0.01, size=signal.shape)
    aug_signals.append(signal + noise)
    if label != ANNOTATION_MAP['N']:
        shift = np.random.randint(-30, 30)
        aug_signals.append(np.roll(signal, shift))
    return aug_signals

# --- Extract beat segments using QRS neighbours ---
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

# --- Dataset generation ---
def build_dataset():
    X, y = [], []
    records = [f[:-4] for f in os.listdir(MITDB_PATH) if f.endswith('.dat')]
    for rec in tqdm(records):
        path = os.path.join(MITDB_PATH, rec)
        try:
            beats, labels = extract_beats(path)
            for beat, label in zip(beats, labels):
                for aug in augment_1d(beat, label):
                    X.append(aug)
                    y.append(label)
        except:
            continue
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y

X, y = build_dataset()
y_cat = to_categorical(y, num_classes=len(ANNOTATION_MAP))

# --- Model architecture ---
def build_model():
    model = Sequential()
    model.add(Input(shape=(WINDOW_SIZE, 1)))
    model.add(Conv1D(64, 5, activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 5, activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 5, activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 5, activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(ANNOTATION_MAP), activation='softmax'))

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Training and evaluation ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold = 1
for train_idx, test_idx in skf.split(X, y):
    print(f"\n--- Fold {fold} ---")
    model = build_model()

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_cat[train_idx], y_cat[test_idx]

    log_path = f'training_log_fold_{fold}.csv'
    checkpoint_path = f'model_fold_{fold}.keras'
    csv_logger = CSVLogger(log_path)
    es = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[es, csv_logger, mc], verbose=1)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(f"\nClassification Report for Fold {fold}:")
    print(classification_report(y_true, y_pred, target_names=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))]))
    plot_conf_matrix(y_true, y_pred, fold)

    fold += 1
