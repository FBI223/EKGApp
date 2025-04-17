import os

import numpy as np

SNOMED_CLASSES = [
    270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002,
    39732003, 164909002, 251146004, 698252002, 10370003, 284470004, 427172004,
    164947007, 111975006, 164917005, 47665007, 59118001, 427393009, 426177001,
    426783006, 427084000, 63593006, 164934002, 59931005, 17338001
]
SNOMED2IDX = {code: i for i, code in enumerate(SNOMED_CLASSES)}

# === ONLY Y DEBUG ===
def debug_y_labels(y_array):
    print("\n[DEBUG] Classes in dataset:")
    for idx, code in enumerate(SNOMED_CLASSES):
        count = int(np.sum(y_array[:, idx]))
        if count > 0:
            print(f"  Class {code}: {count} samples")

    print("\n[DEBUG] Example label rows with active classes:")
    for i in range(min(10, len(y_array))):
        active = [SNOMED_CLASSES[j] for j, v in enumerate(y_array[i]) if v == 1]
        print(f"Sample {i}: {active}")

# === LOAD Y ONLY ===
if os.path.exists('Y.npy'):
    Y = np.load('Y.npy').astype(np.uint8)
    debug_y_labels(Y)
else:
    print("Y.npy not found. Generate dataset first.")