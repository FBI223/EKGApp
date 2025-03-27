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

from main_model.augmentation import augment_chain, AUGMENTATION_FUNCS
from main_model.consts import FS, MITDB_PATH, INV_ANNOTATION_MAP, ANNOTATION_MAP, WINDOW_SIZE, FOLDS, EPOCHS, \
    BATCH_SIZE, RECORD_PATH, MODEL_PATH
from estimator import predict

import tensorflow as tf





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
def augment_1d(signal, label, max_chain_len=len(AUGMENTATION_FUNCS)):
    aug_signals = [signal]

    if label != ANNOTATION_MAP['N']:
        num_augs = random.randint(2, 3)
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


def train_model():

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    X, y = build_dataset()
    y_cat = to_categorical(y, num_classes=len(ANNOTATION_MAP))


    # --- Training and evaluation ---
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    fold = 1
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        print(f"\n--- Fold {fold} ---")
        model = build_model()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]

        log_dir = f"logs/fold_{fold}"
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            CSVLogger(f"{log_dir}/training_log.csv"),
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ModelCheckpoint(f"model_fold_{fold}.keras", monitor='val_accuracy', save_best_only=True),
            TensorBoard(log_dir=log_dir)
        ]

        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=callbacks, verbose=1)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        print(f"\nClassification Report for Fold {fold}:")
        print(classification_report(y_true, y_pred, target_names=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))]))

        plot_conf_matrix(y_true, y_pred, fold)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        fold += 1

    print("\n==== GLOBAL CLASSIFICATION REPORT ====")
    print(classification_report(all_y_true, all_y_pred, target_names=[INV_ANNOTATION_MAP[i] for i in range(len(ANNOTATION_MAP))]))
    plot_conf_matrix(all_y_true, all_y_pred, "global")



def main():
    #predict(RECORD_PATH, MODEL_PATH)
    #train_model()
    print("✅ TF Version:", tf.__version__)
    print("✅ GPU Devices:", tf.config.list_physical_devices('GPU'))



if __name__ == "__main__":
    main()


