import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat
from scipy.signal import resample, butter, sosfiltfilt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
# https://moody-challenge.physionet.org/2020/results/

# https://moody-challenge.physionet.org/2020/results/

# Ścieżki
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'C:\Users\msztu\Documents\EKG_DB_2\challenge2020_data\training'
EXCLUDED_DIRS = ["ptb", "st_petersburg_incart"]
SAVE_DIR = os.path.join(BASE_DIR, 'runtime')
x_path = os.path.join(BASE_DIR, 'X.npy')
y_path = os.path.join(BASE_DIR, 'Y.npy')

# Parametry sygnału
FS_TARGET = 500
WINDOW_SEC = 10
SEG_LENGTH = FS_TARGET * WINDOW_SEC
LEADS = 12
MIN_SEC = 8

# Klasy
SNOMED_IMAGE_CLASSES = {
    '270492004': 'IAVB',
    '164889003': 'AF',
    '164890007': 'AFL',
    '426627000': 'Brady',
    '713427006': 'CRBBB',
    '713426002': 'IRBBB',
    '445118002': 'LAnFB',
    '39732003':  'LAD',
    '164909002': 'LBBB',
    '251146004': 'LQRSV',
    '698252002': 'NSIVCB',
    '10370003':  'PR',
    '284470004': 'PAC',
    '427172004': 'PVC',
    '164947007': 'LPR',
    '111975006': 'LQT',
    '164917005': 'QAb',
    '47665007':  'RAD',
    '59118001':  'RBBB',
    '427393009': 'SA',
    '426177001': 'SB',
    '426783006': 'NSR',
    '427084000': 'STach',
    '63593006':  'SVPB',
    '164934002': 'TAb',
    '59931005':  'TInv',
    '17338001':  'VPB'
}

# Filtracja sygnału
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS_TARGET, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    y = sosfiltfilt(sos, data, axis=1)
    return y

# Wyszukiwanie plików .hea
def list_hea_files(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        if any(ex_dir in root for ex_dir in EXCLUDED_DIRS):
            continue
        for f in files:
            if f.endswith('.hea'):
                paths.append(os.path.join(root, f[:-4]))
    return paths

# Wczytywanie kodów SNOMED
def load_snomed_codes(hea_path):
    codes = []
    with open(hea_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_norm = line.strip().replace('# Dx:', '#Dx:')
        if line_norm.startswith('#Dx:'):
            code_str = line_norm.split(':', 1)[1].strip()
            for c in code_str.split(','):
                c = c.strip()
                if c in SNOMED_IMAGE_CLASSES:
                    codes.append(c)
            break
    return codes

# Segmentacja sygnału
def process_signal(signal, seg_length=SEG_LENGTH, fs_target=FS_TARGET):
    n_samples = signal.shape[1]
    segments = []
    min_samples = MIN_SEC * fs_target
    if n_samples < min_samples:
        return segments
    if n_samples < seg_length:
        padded = np.pad(signal, ((0, 0), (0, seg_length - n_samples)), mode='constant')
        segments.append(padded)
        return segments
    num_full = n_samples // seg_length
    for i in range(num_full):
        seg = signal[:, i*seg_length:(i+1)*seg_length]
        segments.append(seg)
    remainder = n_samples % seg_length
    if remainder >= min_samples:
        last_seg = signal[:, num_full*seg_length:]
        last_seg = np.pad(last_seg, ((0, 0), (0, seg_length - remainder)), mode='constant')
        segments.append(last_seg)
    return segments

# Przygotowanie i zapis danych – uwzględnia augmentację i normalizację (wzór: padding, segmentacja)
def prepare_and_save_data_and_labels(data_dir, fs_target=FS_TARGET, window_sec=WINDOW_SEC, leads=LEADS):
    seg_length = fs_target * window_sec
    max_duration = 30 * fs_target
    metadata = []
    segments = []
    labels_list = []
    file_counter = 0
    hea_files = list_hea_files(data_dir)
    classes = list(SNOMED_IMAGE_CLASSES.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    for base_path in hea_files:
        hea_path = base_path + '.hea'
        mat_path = base_path + '.mat'
        if not (os.path.exists(hea_path) and os.path.exists(mat_path)):
            continue
        codes = load_snomed_codes(hea_path)
        if not codes:
            continue
        with open(hea_path, 'r') as f:
            header_line = f.readline().split()
        if len(header_line) < 4:
            continue
        try:
            n_leads = int(header_line[1])
            fs_orig = float(header_line[2])
        except:
            continue
        if n_leads != leads:
            continue
        try:
            mat_data = loadmat(str(mat_path))
        except Exception:
            continue
        if 'val' not in mat_data:
            continue
        signal = mat_data['val']
        if fs_orig != fs_target:
            target_len = int(signal.shape[1] * fs_target / fs_orig)
            signal = resample(signal, target_len, axis=1)
        if signal.shape[1] > max_duration:
            continue
        signal = butter_bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=fs_target, order=4)
        segments_batch = process_signal(signal, seg_length, fs_target)
        if not segments_batch:
            continue
        y = mlb.transform([list(set(codes))])[0]
        for seg in segments_batch:
            segments.append(seg.astype(np.float32))
            labels_list.append(y.astype(np.float32))
            metadata.append({'file': f'seg_{file_counter}.npy', 'labels': list(np.where(y == 1)[0])})
            file_counter += 1
    np.save(os.path.join(BASE_DIR, '`X.`npy'), np.stack(segments))
    np.save(os.path.join(BASE_DIR, 'Y.npy'), np.stack(labels_list))
    return file_counter

def load_data_into_memory():
    X = np.load(os.path.join(BASE_DIR, 'X.npy'))
    Y = np.load(os.path.join(BASE_DIR, 'Y.npy'))
    return X, Y

# Generator do tf.data.Dataset (zapobiega kopiowaniu całych macierzy do GPU)
def generator(X, Y):
    for i in range(len(X)):
        yield X[i], Y[i]

def create_tf_dataset(X, Y, augment=False, batch_size=32, shuffle=False):
    output_signature = (
        tf.TensorSpec(shape=(LEADS, SEG_LENGTH), dtype=tf.float32),
        tf.TensorSpec(shape=(len(SNOMED_IMAGE_CLASSES),), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(lambda: generator(X, Y), output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if augment:
        # Augmentacja: dodanie szumu Gaussowskiego
        def add_noise(x, y):
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01)
            return x + noise, y
        dataset = dataset.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# -------------------- ULEPSZENIA --------------------
# 1. Custom weighted binary cross-entropy loss
# L = -(1/N)*sum_i sum_c [ w_c (y_ic log(p_ic) + (1-y_ic) log(1-p_ic)) ]
def weighted_binary_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        eps = 1e-8
        loss_val = - (y_true * tf.math.log(y_pred + eps) + (1 - y_true) * tf.math.log(1 - y_pred + eps))
        loss_val = loss_val * weights  # element-wise mnożenie
        return tf.reduce_mean(loss_val)
    return loss

# 2. Funkcja optymalizacji progów decyzyjnych:
# theta_c = argmax_theta F1_c(theta)
def optimize_thresholds(model, X_val, Y_val, steps=101):
    thresholds = np.linspace(0, 1, steps)
    best_thresholds = np.zeros(Y_val.shape[1])
    y_val_pred = model.predict(X_val)
    for c in range(Y_val.shape[1]):
        best_f1 = 0
        best_t = 0
        for t in thresholds:
            y_pred_bin = (y_val_pred[:, c] >= t).astype(int)
            f1 = f1_score(Y_val[:, c], y_pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = best_t
    return best_thresholds

# 3. L2 Regularizacja dodana do warstw (kernel_regularizer)
l2_reg = tf.keras.regularizers.l2(0.001)

# -------------------- MODEL --------------------
class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.attn = tf.keras.layers.Dense(1, kernel_regularizer=l2_reg)
    def call(self, lstm_out):
        attn_weights = tf.math.tanh(self.attn(lstm_out))
        attn_weights = tf.nn.softmax(attn_weights, axis=1)
        context = tf.reduce_sum(attn_weights * lstm_out, axis=1)
        return context

class CNN_LSTM_Attention(tf.keras.Model):
    def __init__(self, in_channels=LEADS, num_classes=None, **kwargs):
        super(CNN_LSTM_Attention, self).__init__(**kwargs)
        if num_classes is None:
            num_classes = len(SNOMED_IMAGE_CLASSES)
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=7, padding='same', kernel_regularizer=l2_reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling1D(pool_size=2)
        ])
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', kernel_regularizer=l2_reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling1D(pool_size=2)
        ])
        self.conv3 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(256, kernel_size=3, padding='same', kernel_regularizer=l2_reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling1D(pool_size=2)
        ])
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True,
                                                                          kernel_regularizer=l2_reg))
        self.attention = Attention(hidden_dim=512)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='sigmoid', kernel_regularizer=l2_reg)
        ])
    def call(self, x, training=False):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        lstm_out = self.bi_lstm(x, training=training)
        context = self.attention(lstm_out)
        out = self.fc(context, training=training)
        return out

# 4. Funkcja normalizacji oceny:
# s_norm = (s_model - s_inactive) / (s_true - s_inactive)
def normalized_score(s_model, s_inactive, s_true):
    return (s_model - s_inactive) / (s_true - s_inactive)

# 5. Przykład k-fold cross walidacji:
def k_fold_train(X, Y, k=5, epochs=10, batch_size=32):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    scores = []
    for train_index, val_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, Y_train = X[train_index], Y[train_index]
        X_val, Y_val = X[val_index], Y[val_index]
        train_ds = create_tf_dataset(X_train, Y_train, augment=True, batch_size=batch_size, shuffle=True)
        val_ds = create_tf_dataset(X_val, Y_val, augment=False, batch_size=batch_size, shuffle=False)
        model = CNN_LSTM_Attention(in_channels=LEADS, num_classes=len(SNOMED_IMAGE_CLASSES))
        model.build(input_shape=(None, LEADS, SEG_LENGTH))
        # Przykładowe wagi klas (można obliczyć na podstawie dystrybucji)
        class_weights = np.ones(len(SNOMED_IMAGE_CLASSES))
        loss_fn = weighted_binary_crossentropy(class_weights)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
        loss, acc = model.evaluate(val_ds)
        print(f"Fold {fold} - Loss: {loss:.4f}, Accuracy: {acc:.4f}\n")
        scores.append(acc)
        fold += 1
    print("Średnia dokładność k-fold:", np.mean(scores))

if __name__ == '__main__':
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("Przetwarzam dane i zapisuję segmenty na dysku...")
        num_segments = prepare_and_save_data_and_labels(DATA_DIR)
        print(f"Zapisano {num_segments} segmentów.")
    else:
        print("Dane już przygotowane.")

    print("Wczytuje wszystkie segmenty do pamięci RAM...")
    X, Y = load_data_into_memory()
    print(f"Kształt X: {X.shape}, Kształt Y: {Y.shape}")

    # Podział danych
    classes = list(SNOMED_IMAGE_CLASSES.keys())
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, train_size=0.8, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    train_dataset = create_tf_dataset(X_train, Y_train, augment=True, batch_size=32, shuffle=True)
    val_dataset = create_tf_dataset(X_val, Y_val, augment=False, batch_size=32, shuffle=False)
    test_dataset = create_tf_dataset(X_test, Y_test, augment=False, batch_size=32, shuffle=False)

    num_classes = len(classes)
    model = CNN_LSTM_Attention(in_channels=LEADS, num_classes=num_classes)
    model.build(input_shape=(None, LEADS, SEG_LENGTH))
    # Użycie wag klas w custom loss (tutaj jednolite, można wyliczyć na podstawie danych)
    class_weights = np.ones(num_classes)
    loss_fn = weighted_binary_crossentropy(class_weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss_fn,
                  metrics=['accuracy'])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "model_epoch_{epoch}.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    num_epochs = 20
    model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, callbacks=[checkpoint_callback])
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("\nTest Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Optymalizacja progów decyzyjnych (po treningu)
    thresholds = optimize_thresholds(model, X_val, Y_val)
    print("Optymalne progi decyzyjne:", thresholds)

    # Przykład k-fold cross walidacji
    # k_fold_train(X, Y, k=5, epochs=5, batch_size=32)
