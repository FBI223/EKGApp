import os
import numpy as np
import wfdb
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Parameters
fs = 360
lowcut, highcut = 0.5, 40.0
pre_sec, post_sec = 0.3, 0.6  # seconds before and after R peak

# Bandpass filter
def bandpass_filter(sig, fs, lowcut, highcut, order=5):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, sig)

# Data loading and segmentation
data_dir = "mitdb"
records = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]
X, y = [], []
label_map = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}  # example mapping

for rec in records:
    rec_path = os.path.join(data_dir, rec)
    signals, _ = wfdb.rdsamp(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    sig = bandpass_filter(signals[:, 0], fs, lowcut, highcut)
    pre = int(pre_sec * fs)
    post = int(post_sec * fs)
    for idx, r in enumerate(ann.sample):
        if r - pre >= 0 and r + post < len(sig):
            beat = sig[r - pre: r + post]
            X.append(beat)
            y.append(label_map.get(ann.symbol[idx], 0))

X = np.array(X)[..., np.newaxis]  # shape: (samples, timesteps, 1)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Squeeze-and-Excitation block
def se_block(inputs, reduction=16):
    channels = inputs.shape[-1]
    se = layers.GlobalAveragePooling1D()(inputs)
    se = layers.Dense(channels // reduction, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, channels))(se)
    return layers.multiply([inputs, se])

# Model: CNN-LSTM-SE
inp = layers.Input(shape=X_train.shape[1:])
x = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(2)(x)
x = se_block(x)
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(len(np.unique(y)), activation='softmax')(x)

model = models.Model(inp, out)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mc = callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Training
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=[es, mc])

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
ticks = np.arange(len(np.unique(y)))
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Save final model
model.save('final_model.h5')
