import os
import random
import numpy as np
import wfdb
from scipy.signal import butter, sosfilt
from tqdm import tqdm

from main_model_beat.consts import (
    FS,
    NOISE_STD_DEFAULT,
    DRIFT_STD,
    DRIFT_FREQ,
    EMG_BAND,
    SHIFT_RANGE,
    AMPLITUDE_SCALE_RANGE,
    WARP_FACTOR_RANGE,
    SPIKE_NUM_DEFAULT,
    SPIKE_STRENGTH,
    DROP_SEGMENT_MIN,
    DROP_SEGMENT_MAX, ANNOTATION_MAP, WINDOW_SIZE, DB_PATHS
)
from main_model_beat.main_keras import extract_beats_with_resampling


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
    X = (X - np.mean(X)) / np.std(X)

    return X[..., np.newaxis], y








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


# --- Gaussian noise ---
def add_gaussian_noise(signal, std=NOISE_STD_DEFAULT):
    noise = np.random.normal(0, std, size=signal.shape)
    return signal + noise

# --- Baseline wander (low freq drift) ---
def add_baseline_wander(signal, std=DRIFT_STD, drift_freq=DRIFT_FREQ, fs=FS):
    noise = np.random.normal(0, std, size=signal.shape)
    sos = butter(2, drift_freq / (fs / 2), btype='low', output='sos')
    drift = sosfilt(sos, noise)
    return signal + drift

# --- EMG noise (high freq bandpass 50–100 Hz) ---
def add_emg_noise(signal, std=NOISE_STD_DEFAULT, fs=FS):
    noise = np.random.normal(0, std, size=signal.shape)
    sos = butter(4, [EMG_BAND[0] / (fs / 2), EMG_BAND[1] / (fs / 2)], btype='bandpass', output='sos')
    emg = sosfilt(sos, noise)
    return signal + emg

# --- Time shifting ---
def time_shift(signal, shift=None):
    if shift is None:
        shift = np.random.randint(SHIFT_RANGE[0], SHIFT_RANGE[1])
    return np.roll(signal, shift)

# --- Amplitude scaling ---
def scale_amplitude(signal, scale_range=AMPLITUDE_SCALE_RANGE):
    scale = np.random.uniform(*scale_range)
    return signal * scale

# --- Time warping (resampling) ---
def time_warp(signal, factor_range=WARP_FACTOR_RANGE):
    factor = np.random.uniform(*factor_range)
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, int(len(signal) * factor))
    warped = np.interp(x_old, x_new[:len(x_old)], signal[:len(x_new)])
    return warped if len(warped) == len(signal) else np.interp(
        np.linspace(0, 1, len(signal)), np.linspace(0, 1, len(warped)), warped)

# --- Impulsive spike ---
def add_spike_noise(signal, num_spikes=SPIKE_NUM_DEFAULT, spike_strength=SPIKE_STRENGTH):
    signal = signal.copy()
    for _ in range(num_spikes):
        idx = np.random.randint(0, len(signal))
        signal[idx] += np.random.choice([-1, 1]) * spike_strength
    return signal

# --- Drop segment ---
def drop_segment(signal, min_fraction=DROP_SEGMENT_MIN, max_fraction=DROP_SEGMENT_MAX):
    signal = signal.copy()
    L = len(signal)
    drop_len = int(np.random.uniform(min_fraction, max_fraction) * L)
    start = np.random.randint(0, L - drop_len)
    signal[start:start + drop_len] = 0
    return signal

# --- Lista funkcji augmentujących ---
AUGMENTATION_FUNCS = [
    add_gaussian_noise,
    add_baseline_wander,
    add_emg_noise,
    time_shift,
    scale_amplitude,
    time_warp,
    add_spike_noise,
    drop_segment
]


# --- Augmentacja łańcuchowa (losowy zestaw funkcji) ---
def augment_chain(signal, funcs=None, max_chain_len=None):
    """
    Losuje i stosuje losowy zestaw funkcji augmentujących (1 do N).
    """
    if funcs is None:
        funcs = AUGMENTATION_FUNCS
    if max_chain_len is None:
        max_chain_len = len(funcs)
    k = random.randint(1, max_chain_len)
    chosen_funcs = random.sample(funcs, k)
    aug_signal = signal.copy()
    for func in chosen_funcs:
        aug_signal = func(aug_signal)
    return aug_signal


# --- Augmentation ---
def augment_1d(signal, label, max_chain_len=len(AUGMENTATION_FUNCS)):
    aug_signals = [signal]

    if label != ANNOTATION_MAP['N']:
        num_augs = random.randint(3, 5)
        for _ in range(num_augs):
            aug = augment_chain(signal, AUGMENTATION_FUNCS, max_chain_len)
            aug_signals.append(aug)
    return aug_signals
