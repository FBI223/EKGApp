FS=360
MITDB_PATH = '../databases/mitdb'
SVDB_PATH = '../databases/svdb'
WINDOW_SIZE = FS
FOLDS = 4
EPOCHS = 8
BATCH_SIZE = 16



# --- Annotation map ---
ANNOTATION_MAP = {
    'N': 0, 'V': 1, '/': 2, 'R': 3, 'L': 4, 'A': 5, '!': 6, 'E': 7
}
INV_ANNOTATION_MAP = {v: k for k, v in ANNOTATION_MAP.items()}









## Szum Gaussa
NOISE_STD_DEFAULT = 0.02              # silniejszy szum

## Baseline wander
DRIFT_STD = 0.05                       # wyraźniejszy dryft
DRIFT_FREQ = 0.2                       # wolny dryft oddechowy

## EMG noise (szum mięśniowy)
EMG_BAND = (40, 100)                   # szersze pasmo EMG

# --- Transformacje czasowe i amplitudowe ---
SHIFT_RANGE = (-30, 30)                # większe przesunięcia
AMPLITUDE_SCALE_RANGE = (0.8, 1.2)     # wyraźniejsze skalowanie
WARP_FACTOR_RANGE = (0.85, 1.15)       # mocniejsze deformacje czasu

# --- Szpilki i segmenty ---
SPIKE_NUM_DEFAULT = 2                 # więcej impulsów
SPIKE_STRENGTH = 0.15                 # wyraźniejsze zakłócenie

DROP_SEGMENT_MIN = 0.02               # bardziej widoczne wycięcia
DROP_SEGMENT_MAX = 0.07               # większy zakres usuwanych segmentów
