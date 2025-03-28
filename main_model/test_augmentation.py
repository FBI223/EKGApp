import wfdb
import numpy as np
import matplotlib.pyplot as plt

from main_model.augmentation import (
    add_gaussian_noise, add_emg_noise, add_baseline_wander
)
from consts import WINDOW_SIZE, NOISE_STD_DEFAULT, DRIFT_STD
from main_model.deaugmentation import denoise_pipeline


def extract_beats_from_mitdb(record_path, num_beats=5, window_size=WINDOW_SIZE):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]
    fs = record.fs

    beats = []
    for i in range(1, len(annotation.sample) - 1):
        prev = annotation.sample[i - 1] + 20
        next_ = annotation.sample[i + 1] - 20
        if prev < next_ and next_ <= len(signal):
            segment = signal[prev:next_]
            if len(segment) != window_size:
                segment = np.interp(np.linspace(0, len(segment)-1, window_size),
                                    np.arange(len(segment)), segment)
            beats.append(segment)
        if len(beats) >= num_beats:
            break
    return beats


def plot_comparison(original, noisy, denoised, title_prefix='', idx=0):
    plt.figure(figsize=(14, 4))
    plt.plot(original, label='Oryginał', lw=1.5)
    plt.plot(noisy, label='Zaszumiony', lw=1.2, alpha=0.7)
    plt.plot(denoised, label='Oczyszczony', lw=1.5, linestyle='--')
    plt.title(f"{title_prefix} – beat {idx}")
    plt.xlabel("Próbki")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



record_path = '../databases/mitdb/108'  # ścieżka bez .dat
beats = extract_beats_from_mitdb(record_path, num_beats=10)

# --- Augmentacje: można dowolnie zmieniać ---
for i, beat in enumerate(beats):
    # Dodaj zakłócenia (przykład: Gaussian + EMG + Baseline wander)
    noisy = add_gaussian_noise(beat, std=NOISE_STD_DEFAULT)
    noisy = add_emg_noise(noisy, std=NOISE_STD_DEFAULT)
    noisy = add_baseline_wander(noisy, std=DRIFT_STD)

    # Oczyszczanie
    filtered = clean = denoise_pipeline(noisy)

    # Wyświetlenie porównania
    plot_comparison(beat, noisy, filtered, title_prefix='Porównanie', idx=i+1)


