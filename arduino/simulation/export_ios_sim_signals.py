"""
Export the ESP32 simulator signals (ekgXXX_5min_signal.h) as JSON files
for the iOS app's built-in BLE device simulation.

Usage:  python3 export_ios_sim_signals.py
Output: app_ios/EKGApp/EKGApp/SimulationSignals/sim_<record>.json
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent.parent / "app_ios" / "EKGApp" / "EKGApp" / "SimulationSignals"


def parse_header(path: Path):
    text = path.read_text()
    rec = re.search(r"ekg(\d+)_5min\[\]", text).group(1)
    body = re.search(r"\{(.*?)\}", text, re.S).group(1)
    samples = [int(v) for v in re.findall(r"-?\d+", body)]

    def const(name):
        return int(re.search(rf"EKG{rec}_{name}\s*=\s*(-?\d+)", text).group(1))

    return rec, samples, const("GAIN"), const("BASELINE"), const("FS")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for header in sorted(HERE.glob("ekg*_5min_signal.h")):
        rec, samples, gain, baseline, fs = parse_header(header)
        mv = [round((s - baseline) / gain, 4) for s in samples]
        data = {
            "name": f"SVDB {rec}",
            "description": f"MIT-BIH Supraventricular Arrhythmia DB, record {rec}, lead 0, {len(mv) // fs // 60} min",
            "fs": fs,
            "leads": ["ECG"],
            "signals": [mv],
        }
        out = OUT_DIR / f"sim_{rec}.json"
        out.write_text(json.dumps(data, separators=(",", ":")))
        print(f"✅ {out.name}: {len(mv)} samples @ {fs} Hz")


if __name__ == "__main__":
    main()
