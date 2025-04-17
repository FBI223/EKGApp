import os
import numpy as np
import wfdb
from scipy import signal
from collections import defaultdict

AAMI_CLASSES = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q']
}

def map_annotation_to_aami(symbol):
    for key, values in AAMI_CLASSES.items():
        if symbol in values:
            return key
    return None

def load_mitdb_segmented(path, original_fs=360, target_fs=125, segment_sec=1.2, interpolated_len=187):
    records = [f[:-4] for f in os.listdir(path) if f.endswith('.dat')]
    X, y = [], []

    for record in records:
        try:
            record_path = os.path.join(path, record)
            signal_data, _ = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            sig = signal_data[:, 0]
            duration = len(sig) / original_fs
            num_target_samples = int(duration * target_fs)
            resampled_sig = signal.resample(sig, num_target_samples)

            scale = target_fs / original_fs
            new_peaks = [int(p * scale) for p in annotation.sample]

            segment_len = int(target_fs * segment_sec)
            half_seg = segment_len // 2

            for i, peak in enumerate(new_peaks):
                label = map_annotation_to_aami(annotation.symbol[i])
                if label is None:
                    continue

                start = peak - half_seg
                end = peak + half_seg
                if start < 0 or end > len(resampled_sig):
                    continue

                beat_segment = resampled_sig[start:end]
                interpolated = signal.resample(beat_segment, interpolated_len)
                X.append(interpolated.astype(np.float32))
                y.append(label)

        except Exception as e:
            print(f"Błąd w rekordzie {record}: {e}")

    return np.array(X), np.array(y)

def save_class_to_header(segments, class_label, out_dir="headers"):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"class_{class_label}.h")
    with open(filename, 'w') as f:
        f.write(f"#ifndef CLASS_{class_label}_H\n#define CLASS_{class_label}_H\n\n")
        f.write(f"const float class_{class_label}_samples[50][187] = {{\n")
        for s in segments:
            line = ", ".join([f"{v:.6f}f" for v in s])
            f.write(f"  {{ {line} }},\n")
        f.write("};\n\n")
        f.write(f"#endif // CLASS_{class_label}_H\n")

def main():
    X, y = load_mitdb_segmented("../../databases/mitdb")
    class_data = defaultdict(list)
    for x, label in zip(X, y):
        if len(class_data[label]) < 50:
            class_data[label].append(x)

    for class_label in ['N', 'S', 'V', 'F', 'Q']:
        if len(class_data[class_label]) >= 50:
            save_class_to_header(class_data[class_label][:50], class_label)

if __name__ == "__main__":
    main()
