FS_TARGET=360
FS=360

DB_PATHS = [
    ('../databases/mitdb', 360),
    ('../databases/svdb', 128),
    ('../databases/incartdb', 257)
]

DB_PATHS2 = [
    ('../databases/mitdb', 360),
    ('../databases/svdb', 128),
    ('../databases/incartdb', 257)
]

MITDB_PATH = '../databases/mitdb'
SVDB_PATH = '../databases/svdb'
INCARTDB_PATH = '../databases/incartdb'
NSRDB_PATH = '../databases/nsrdb'
CUDB_PATH = '../databases/cudb'


WINDOW_SIZE = FS_TARGET
FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 16


MODEL_PATH = "trained_models/v3/model_fold_1.keras"
RECORD_PATH_2 = '../databases/mitdb/105'
RECORD_PATH = '../databases/svdb/854'

# --- Annotation map ---
ANNOTATION_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    '/': 3, 'f': 3, 'Q': 3, '!': 3, '|': 3, '~': 3, 'x': 3
}
INV_ANNOTATION_MAP = {
    0: 'N',
    1: 'S',
    2: 'V',
    3: 'Q'
}

ANNOTATION_MAP1 = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}
INV_ANNOTATION_MAP1 = {v: k for k, v in ANNOTATION_MAP.items()}









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