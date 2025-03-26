from scipy.signal import savgol_filter, medfilt
import numpy as np

def denoise_pipeline(signal, fs=360):
    """
    Usuwa baseline wander i EMG bez przesunięcia QRS ani osi Y.
    Gwarantuje dopasowanie średniego poziomu (DC offsetu) do oryginału.
    """
    signal = signal.astype(np.float64)
    original_mean = np.mean(signal)

    # --- baseline wander przez Savitzky-Golay
    baseline = savgol_filter(signal, window_length=301, polyorder=2)
    corrected = signal - baseline

    # --- median filter (na szpilki)
    corrected = medfilt(corrected, kernel_size=3)

    # --- wymuszone wyrównanie poziomu amplitudy
    corrected = corrected - np.mean(corrected) + original_mean

    return corrected
