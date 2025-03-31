import os
import random
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
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
from sklearn.model_selection import train_test_split

from main_model.augmentation import augment_chain, AUGMENTATION_FUNCS
from consts import MITDB_PATH, INV_ANNOTATION_MAP, ANNOTATION_MAP, WINDOW_SIZE, FOLDS, EPOCHS, BATCH_SIZE, DB_PATHS, FS_TARGET
from scipy.signal import resample

import tensorflow as tf





# --- Utility functions ---
def plot_conf_matrix(y_true, y_pred, fold):
    cm = confusion_matrix(y_true, y_pred)
    labels = [INV_ANNOTATION_MAP[i] for i in range(len(INV_ANNOTATION_MAP))]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'conf_matrix_fold_{fold}.png')
    plt.close()


# --- Augmentation ---
def augment_1d(signal, label, max_chain_len=len(AUGMENTATION_FUNCS)):
    aug_signals = [signal]

    if label != ANNOTATION_MAP['N']:
        num_augs = random.randint(1, 3)
        for _ in range(num_augs):
            aug = augment_chain(signal, AUGMENTATION_FUNCS, max_chain_len)
            aug_signals.append(aug)
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
            prev = annotation.sample[i] - WINDOW_SIZE // 2
            next_ = annotation.sample[i ] + (WINDOW_SIZE // 2)
            if prev < next_ and next_ <= len(signal):
                segment = signal[prev:next_].astype(np.float32)
                if len(segment) != WINDOW_SIZE:
                    segment = np.interp(np.linspace(0, len(segment) - 1, WINDOW_SIZE),
                                        np.arange(len(segment)), segment)
                beats.append(segment)
                labels.append(ANNOTATION_MAP[sym])
    return beats, labels



def extract_beats_with_resampling(record_path, fs_orig):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]
    signal_rs, ann_samples_rs = resample_signal_and_annotations(signal, annotation.sample, fs_orig, FS_TARGET)

    beats, labels = [], []
    for i in range(1, len(ann_samples_rs) - 1):
        sym = annotation.symbol[i]
        if sym in ANNOTATION_MAP:
            center = ann_samples_rs[i]
            prev = center - WINDOW_SIZE // 2
            next_ = center + WINDOW_SIZE // 2
            if prev < next_ and next_ <= len(signal_rs):
                segment = signal_rs[prev:next_].astype(np.float32)
                if len(segment) != WINDOW_SIZE:
                    segment = np.interp(np.linspace(0, len(segment) - 1, WINDOW_SIZE),
                                        np.arange(len(segment)), segment)
                beats.append(segment)
                labels.append(ANNOTATION_MAP[sym])
    return beats, labels




def resample_signal_and_annotations(signal, annotation_samples, fs_orig, fs_target):
    resampled_signal = resample(signal, int(len(signal) * fs_target / fs_orig))
    factor = fs_target / fs_orig
    resampled_ann = [int(s * factor) for s in annotation_samples]
    return resampled_signal, resampled_ann


# --- Dataset generation ---
def build_dataset_augmented():
    X, y = [], []
    for db_path, fs in DB_PATHS:
        records = [f[:-4] for f in os.listdir(db_path) if f.endswith('.dat')]
        for rec in tqdm(records, desc=f"Loading {os.path.basename(db_path)}"):
            path = os.path.join(db_path, rec)
            try:
                beats, labels = extract_beats_with_resampling(path, fs)
                for beat, label in zip(beats, labels):
                    if label == ANNOTATION_MAP['N']:
                        X.append(beat)
                        y.append(label)
                    else:
                        for aug in augment_1d(beat, label):
                            X.append(aug)
                            y.append(label)
            except:
                continue
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y




def build_dataset():
    X, y = [], []
    for db_path, fs in DB_PATHS:
        records = [f[:-4] for f in os.listdir(db_path) if f.endswith('.dat')]
        for rec in tqdm(records, desc=f"Loading {os.path.basename(db_path)}"):
            path = os.path.join(db_path, rec)
            try:
                beats, labels = extract_beats_with_resampling(path, fs)
                X.extend(beats)
                y.extend(labels)
            except:
                continue
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y


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
    model.add(Dense(len(INV_ANNOTATION_MAP), activation='softmax'))

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model():
    print("✅ TF Version:", tf.__version__)
    print("✅ GPU Devices:", tf.config.list_physical_devices('GPU'))

    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    #X, y = build_dataset()
    X,y = build_dataset_augmented()
    y_cat = to_categorical(y, num_classes=len(INV_ANNOTATION_MAP))

    target_names = [INV_ANNOTATION_MAP[i] for i in range(len(INV_ANNOTATION_MAP))]

    # --- Simple train/val split ---
    X_train, X_val, y_train, y_val, y_train_raw, y_val_raw = train_test_split(
        X, y_cat, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint("model_final.h5", monitor='val_accuracy', save_best_only=True),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=callbacks, verbose=1)

    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("Unique labels in y_true:", np.unique(y_true))
    print("Unique labels in y_pred:", np.unique(y_pred))

    print(f"\n✅ Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    plot_conf_matrix(y_true, y_pred, fold="val")

def main():
    train_model()


if __name__ == "__main__":
    main()

