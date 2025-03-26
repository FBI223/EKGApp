import os, glob
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from google.colab import drive


drive.mount('/content/drive')
save_path_google = "/content/drive/MyDrive/EKG_data"
os.makedirs(save_path, exist_ok=True)

IMG_SIZE = (128, 128)
CROP_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 50
DB_PATH = "../../databases/mitdb"

CLASS_DICT = {
    "N": 0,  # Normal beat
    "PVC": 1,
    "PAB": 2,
    "RBB": 3,
    "LBB": 4,
    "APC": 5,
    "VFW": 6,
    "VEB": 7
}

RECORDS_BY_CLASS = {
    "N": ["100", "101", "103", "105", "108", "112", "113", "114", "115", "117", "121", "122", "123", "202", "205", "219", "230", "234"],
    "PVC": ["106", "116", "119", "200", "201", "203", "208", "210", "213", "215", "221", "228", "233"],
    "PAB": ["102", "104", "107", "217"],
    "RBB": ["118", "124", "212", "231"],
    "LBB": ["109", "111", "207", "213"],
    "APC": ["209", "220", "222", "223", "232"],
    "VFW": ["207"],
    "VEB": ["207"]
}

ANNOTATION_MAP = {
    "N": "N", "·": "N",
    "V": "PVC",
    "/": "PAB",
    "R": "RBB",
    "L": "LBB",
    "A": "APC", "a": "APC",
    "!": "VFW",
    "E": "VEB"
}

def map_annotation(symbol):
    return ANNOTATION_MAP.get(symbol)

def get_all_records():
    all_records = []
    for recs in RECORDS_BY_CLASS.values():
        all_records.extend(recs)
    return sorted(set(all_records))

def load_segments():
    rec_names = get_all_records()
    segments, labels = [], []
    class_counts = {}
    all_annotations = set()
    for rec in rec_names:
        print(f"=== Przetwarzanie rekordu: {rec} ===")
        record = wfdb.rdrecord(os.path.join(DB_PATH, rec))
        sig = record.p_signal[:, 0]
        ann = wfdb.rdann(os.path.join(DB_PATH, rec), 'atr')

        for i in range(1, len(ann.sample) - 1):
            sym = ann.symbol[i]
            all_annotations.add(sym)
            mapped = map_annotation(sym)
            if mapped is None:
                continue

            start = ann.sample[i - 1] + 20
            end = ann.sample[i + 1] - 20
            if end - start <= 0:
                continue

            segment = sig[start:end]
            segments.append(segment)
            labels.append(mapped)
            class_counts[mapped] = class_counts.get(mapped, 0) + 1

    print("=== Wszystkie unikalne adnotacje ===")
    print(sorted(list(all_annotations)))
    print("=== Rozkład klas ===")
    print(class_counts)
    return segments, labels

def signal_to_img(signal):
    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.axis('off')
    plt.plot(signal, color='black', linewidth=1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.gca().set_xticks([]); plt.gca().set_yticks([])
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = img.reshape((h, w, 3))
    plt.close(fig)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(img_gray, IMG_SIZE)

def augment_img(img):
    crops = []
    h, w = img.shape
    positions = [
        (0, 0), (0, w - CROP_SIZE), (0, (w - CROP_SIZE) // 2),
        ((h - CROP_SIZE) // 2, 0), ((h - CROP_SIZE) // 2, (w - CROP_SIZE) // 2),
        ((h - CROP_SIZE) // 2, w - CROP_SIZE),
        (h - CROP_SIZE, 0), (h - CROP_SIZE, (w - CROP_SIZE) // 2), (h - CROP_SIZE, w - CROP_SIZE)
    ]
    for y, x in positions:
        crop = img[y:y + CROP_SIZE, x:x + CROP_SIZE]
        crop = cv2.resize(crop, IMG_SIZE)
        crops.append(crop)
    return crops

def create_model(input_shape=(128, 128, 1), num_classes=len(CLASS_DICT)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_initializer='glorot_uniform'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, kernel_initializer='glorot_uniform'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def main():
    segments, labels = load_segments()
    X, y = [], []
    num_classes = len(CLASS_DICT)

    print("=== Generowanie obrazów i augmentacja ===")
    for seg, label in zip(segments, labels):
        img = signal_to_img(seg)
        img = img.astype('float32') / 255.0
        X.append(img.reshape(IMG_SIZE[0], IMG_SIZE[1], 1))
        y.append(to_categorical(CLASS_DICT[label], num_classes))

        if label != "N":
            for aug in augment_img(img):
                aug = aug.astype('float32') / 255.0
                X.append(aug.reshape(IMG_SIZE[0], IMG_SIZE[1], 1))
                y.append(to_categorical(CLASS_DICT[label], num_classes))

    X = np.array(X)
    y = np.array(y)
    print(f"Finalny rozmiar X: {X.shape}, y: {y.shape}")

    np.save(os.path.join(save_path_google, "X.npy"), X, allow_pickle=False)
    np.save(os.path.join(save_path_google, "y.npy"), y, allow_pickle=False)
    print(f"Zapisano pliki X.npy oraz y.npy w {save_path_google}")


    # === ZAPISZ DO DYSKU ===
    np.save("X.npy", X, allow_pickle=False)
    np.save("y.npy", y, allow_pickle=False)
    print("Zapisano X.npy oraz y.npy w bieżącym folderze.")



if __name__ == "__main__":
    main()
