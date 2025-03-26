import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
X = np.load(X_FILE)
y = np.load(Y_FILE)

print(f"✅ Załadowano: X.shape={X.shape}, y.shape={y.shape}")

# === Podział na zbiór treningowy i walidacyjny ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    stratify=y.argmax(axis=1),
    test_size=0.2, random_state=42
)

# === Tworzenie datasetów ===
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
    model = create_model()
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy', verbose=1)
    ]

    print("🚀 Start treningu modelu...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Trening zakończony")

    print("📊 Obliczanie metryk na zbiorze walidacyjnym...")
    y_val_pred = model.predict(X_val, batch_size=BATCH_SIZE)
    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_val_true, y_val_pred_classes))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_val_true, y_val_pred_classes)
    print(cm)

    np.save("confusion_matrix.npy", cm)
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(cm).plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    return model

# === Uruchomienie pipeline'u ===
if __name__ == "__main__":
    model = train_model()
