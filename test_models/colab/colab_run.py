import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === Ścieżka do plików lokalnych ===
DATA_PATH = os.getcwd()
X_FILE = os.path.join(DATA_PATH, "X.npy")
Y_FILE = os.path.join(DATA_PATH, "y.npy")

# === Parametry ===
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50

# === Wczytywanie danych ===
print("🔄 Wczytywanie X.npy oraz y.npy...")
X = np.load(X_FILE, mmap_mode='r')
y = np.load(Y_FILE, mmap_mode='r')

print(f"✅ Załadowano: X.shape={X.shape}, y.shape={y.shape}")

# === Definicja modelu ===
def create_model(input_shape=(128, 128, 1), num_classes=y.shape[1]):
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

# === Funkcja trenowania ===
def train_model():
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_argmax = np.argmax(y, axis=1)
    fold = 1
    all_reports = []

    for train_idx, val_idx in kf.split(np.arange(X.shape[0]), y_argmax):
        print(f"\n📁 Fold {fold}...")

        X_train_idx, X_val_idx = train_idx, val_idx
        y_train_idx, y_val_idx = train_idx, val_idx

        def data_generator(indices):
            for idx in indices:
                yield X[idx], y[idx]

        train_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(X_train_idx),
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
                tf.TensorSpec(shape=(y.shape[1],), dtype=tf.float32)
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(X_val_idx),
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
                tf.TensorSpec(shape=(y.shape[1],), dtype=tf.float32)
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = create_model()
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            ModelCheckpoint(f"best_model_fold{fold}.h5", save_best_only=True, monitor='val_accuracy', verbose=1)
        ]

        model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks, verbose=1)

        X_val_array = np.array([X[i] for i in X_val_idx])
        y_val_array = np.array([y[i] for i in y_val_idx])
        y_val_pred = model.predict(X_val_array, batch_size=BATCH_SIZE)
        y_val_true = np.argmax(y_val_array, axis=1)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)

        print(f"\n=== Classification Report for Fold {fold} ===")
        report = classification_report(y_val_true, y_val_pred_classes, output_dict=False)
        print(report)
        all_reports.append(report)

        print(f"\n=== Confusion Matrix for Fold {fold} ===")
        cm = confusion_matrix(y_val_true, y_val_pred_classes)
        print(cm)

        np.save(f"confusion_matrix_fold{fold}.npy", cm)
        plt.figure(figsize=(10, 8))
        ConfusionMatrixDisplay(cm).plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix Fold {fold}")
        plt.savefig(f"confusion_matrix_fold{fold}.png")
        plt.close()

        fold += 1

    model.save("final_model.h5")
    print("📦 Zapisano finalny model jako final_model.h5")
    return model

# === Uruchomienie pipeline'u ===
if __name__ == "__main__":
    model = train_model()