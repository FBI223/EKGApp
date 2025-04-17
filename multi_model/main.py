# ECG 12-lead SE-ResNet34 TensorFlow classification pipeline with evaluation and export (strictly per CinC2020-035)

import os
import numpy as np
import wfdb
import scipy.io
from scipy.signal import resample
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix, roc_auc_score
import seaborn as sns
import datetime
import subprocess
from tqdm import tqdm
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1800)]
        )
    except RuntimeError as e:
        print(e)

# === CONSTANTS ===
BASE_PATH = '/home/msztu223/Documents/ECG/EKG_DB_2/challenge2020_data/training'
FS_TARGET = 500
SEGMENT_SAMPLES = FS_TARGET * 10  # 10 seconds
NUM_LEADS = 12
BATCH_SIZE = 32
EPOCHS = 20
SNOMED_CLASSES = [
    270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002,
    39732003, 164909002, 251146004, 698252002, 10370003, 284470004, 427172004,
    164947007, 111975006, 164917005, 47665007, 59118001, 427393009, 426177001,
    426783006, 427084000, 63593006, 164934002, 59931005, 17338001
]
SNOMED2IDX = {code: i for i, code in enumerate(SNOMED_CLASSES)}


def load_weight_matrix():
    W = np.loadtxt("weights.csv", delimiter=",")
    return W


def compute_challenge_score(y_true, y_pred, W):
    score = 0.0
    for i in range(len(y_true)):
        tp_idx = np.where(y_true[i] == 1)[0]
        pred_idx = np.where(y_pred[i] == 1)[0]
        for t in tp_idx:
            for p in pred_idx:
                score += W[t][p]
    return score / len(y_true)

# === DATA ===
def parse_labels(header_path):
    with open(header_path, 'r') as f:
        lines = f.readlines()
    label = np.zeros(len(SNOMED_CLASSES), dtype=np.float32)
    for l in lines:
        if l.startswith('# Dx:'):
            codes = list(map(int, l.strip().split(': ')[1].split(',')))
            for code in codes:
                if code in SNOMED2IDX:
                    label[SNOMED2IDX[code]] = 1
    return label

def parse_demographics(header_path):
    age, sex = 0, 0
    with open(header_path, 'r') as f:
        for line in f:
            if line.startswith('# Age:'):
                try:
                    age = int(line.split(': ')[1].strip())
                except: pass
            elif line.startswith('# Sex:'):
                sex = 1 if line.split(': ')[1].strip().upper() == 'MALE' else 0
    return [age / 100.0, sex]

def load_record(path_no_ext):
    if 'incart' in path_no_ext.lower():  # Exclude INCART database
        return []
    try:
        record = wfdb.rdrecord(path_no_ext)
        signal = record.p_signal.T
        fs = record.fs
        if signal.shape[0] != NUM_LEADS:
            return []
        signal = resample(signal, int(signal.shape[1] * FS_TARGET / fs), axis=1)
        if signal.shape[1] < SEGMENT_SAMPLES:
            pad_width = SEGMENT_SAMPLES - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)))
        else:
            signal = signal[:, :SEGMENT_SAMPLES]
        label = parse_labels(path_no_ext + '.hea')
        demo = parse_demographics(path_no_ext + '.hea')
        return [(signal, label, demo)]
    except:
        return []

def load_all_data():
    signals, labels, demos = [], [], []
    all_files = [os.path.join(subdir, f[:-4]) for subdir, _, files in os.walk(BASE_PATH) for f in files if f.endswith('.mat')]
    for base in tqdm(all_files, desc="Loading ECG data"):
        for seg, lab, demo in load_record(base):
            signals.append(seg)
            labels.append(lab)
            demos.append(demo)
    return np.array(signals), np.array(labels), np.array(demos)

# === MODEL ===
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    se = tf.keras.layers.Dense(filters // reduction, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    return tf.keras.layers.Multiply()([input_tensor, tf.keras.layers.Reshape((1, filters))(se)])

def residual_block(x, filters, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, 3, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)

def resnet_block_with_se(x, filters, blocks, stride=1):
    x = residual_block(x, filters, stride)
    for _ in range(1, blocks):
        x = residual_block(x, filters)
    return x

def build_se_resnet34(input_shape=(SEGMENT_SAMPLES, NUM_LEADS), demo_shape=(2,), num_classes=len(SNOMED_CLASSES)):
    ecg_input = tf.keras.Input(shape=input_shape)
    demo_input = tf.keras.Input(shape=demo_shape)

    x = tf.keras.layers.Conv1D(64, 7, strides=2, padding='same')(ecg_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = resnet_block_with_se(x, 64, 3)
    x = resnet_block_with_se(x, 128, 4, stride=2)
    x = resnet_block_with_se(x, 256, 6, stride=2)
    x = resnet_block_with_se(x, 512, 3, stride=2)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, demo_input])
    out = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    return tf.keras.Model(inputs=[ecg_input, demo_input], outputs=out)

# === TRAINING ===
if os.path.exists('X.npy') and os.path.exists('Y.npy') and os.path.exists('D.npy'):
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    D = np.load('D.npy')
else:
    X, Y, D = load_all_data()
    X = np.transpose(X, (0, 2, 1))  # (samples, time, leads)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    np.save('D.npy', D)

label_counts = np.sum(Y, axis=0)
print("Label distribution:")
for code, count in zip(SNOMED_CLASSES, label_counts):
    print(f"{code}: {int(count)}")

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]
D_train, D_val = D[:split], D[split:]

train_ds = tf.data.Dataset.from_tensor_slices(((X_train, D_train), Y_train)).shuffle(2048).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices(((X_val, D_val), Y_val)).batch(BATCH_SIZE)

model = build_se_resnet34()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
subprocess.Popen(["tensorboard", "--logdir", log_dir])
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tensorboard_cb
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# === EVAL ===
W = load_weight_matrix()
preds = model.predict(val_ds)
preds_binary = (preds > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(Y_val, preds_binary, zero_division=0))

try:
    print("Macro AUC:", roc_auc_score(Y_val, preds, average='macro'))
except:
    print("AUC failed")

challenge_score = compute_challenge_score(Y_val, preds_binary, W)
print(f"Challenge Score: {challenge_score:.4f}")

cm = multilabel_confusion_matrix(Y_val, preds_binary)
for i, code in enumerate(SNOMED_CLASSES):
    tn, fp, fn, tp = cm[i].ravel()
    print(f"Class {code} — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {code}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# === EXPORT ===
model.save('ecg_se_resnet34_saved_model')
converter = tf.lite.TFLiteConverter.from_saved_model('ecg_se_resnet34_saved_model')
tflite_model = converter.convert()
with open("ecg_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved as ecg_model.tflite")
