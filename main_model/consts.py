FS_TARGET=128
FS=128
WINDOW_SIZE = 188
FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


DB_PATHS = [
    ('../databases/mitdb', 360),
    ('../databases/svdb', 128),
    ('../databases/incartdb', 257)
]


MITDB_PATH = '../databases/mitdb'
SVDB_PATH = '../databases/svdb'
INCARTDB_PATH = '../databases/incartdb'
NSRDB_PATH = '../databases/nsrdb'




# --- Annotation map ---
ANNOTATION_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular (S)
    'V': 2, 'E': 2,                          # Ventricular (V)
    'F': 3,                                  # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4, '!': 4, '|': 4, '~': 4, 'x': 4  # Unknown (Q)
}
INV_ANNOTATION_MAP = {
    0: 'N',
    1: 'S',
    2: 'V',
    3: 'F',
    4: 'Q'
}


NUM_CLASSES = len(INV_ANNOTATION_MAP)









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