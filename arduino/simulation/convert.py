import wfdb
import numpy as np

# --- Pobranie rekordu od użytkownika ---
rec_name = input("🔢 Podaj nazwę rekordu (np. 820): ").strip()

# --- Odczyt rekordu i nagłówka ---
record = wfdb.rdrecord(rec_name, pn_dir="svdb")
signal = record.p_signal[:, 0]  # tylko pierwszy kanał (lead 0)

# --- Parametry z nagłówka ---
gain = int(record.adc_gain[0])         # gain leadu 0
baseline = int(record.baseline[0])     # baseline leadu 0
fs = int(record.fs)                    # sampling frequency

# --- Wytnij dokładnie 5 minut ---
target_len = fs * 60 * 5               # 5 minut
lead = signal[:target_len]             # przytnij
length = len(lead)

# --- Konwersja do ADU (int16_t) ---
adu = (lead * gain + baseline).astype("int16")

# --- Nazwa pliku wynikowego ---
header_name = f"ekg{rec_name}_5min_signal.h"

# --- Zapis .h ---
with open(header_name, "w") as f:
    f.write("#pragma once\n\n")
    f.write(f"const int16_t ekg{rec_name}_5min[] = {{\n")
    for i, val in enumerate(adu):
        f.write(f"{val}, ")
        if i % 10 == 9:
            f.write("\n")
    f.write("};\n\n")
    f.write(f"const size_t ekg{rec_name}_5min_len = {length};\n")
    f.write(f"const int EKG{rec_name}_GAIN = {gain};\n")
    f.write(f"const int EKG{rec_name}_BASELINE = {baseline};\n")
    f.write(f"const int EKG{rec_name}_FS = {fs};\n")

print(f"✅ Zapisano: {header_name}")
