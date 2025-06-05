



# 📱 ECGApp – Mobile ECG Viewer & Classifier

**ECGApp** is a cross-platform mobile application (iOS & Android) for real-time ECG signal acquisition, processing, and classification using deep learning models.

---

## 🧠 Key Features

* 🔌 Connect to external BLE ECG sensors (e.g., ESP32, Arduino)
* 📈 Real-time signal visualization (zoom, pan, rescale)
* ⏺ Record ECG to local `.json` format
* 🧠 AI-based classification:

  * Heart **rhythm classification** (e.g. NSR, AF, PVC)
  * **Beat-type classification** near QRS centers (e.g. N, V, S)
* 📂 Browse, view, share or delete saved recordings

---

## 📦 Used Deep Learning Models

### 1. Rhythm Classification – `SE_MobileNet1D_noLSTM` (PyTorch)

Lightweight SE-enhanced MobileNet 1D CNN with demographic inputs.

* **Input**: 10-second 1-lead ECG (5000 samples of 500hz signal), age (normalized), sex (0/1)
* **Output**: Single-label rhythm class (softmax)
* **Deployment**: Converted via ONNX → TensorFlow → CoreML / TFLite
* **Datasets**: CPSC 2018, CPSC Extra, PTB-XL, Georgia Dataset

#### Architecture Summary:

```
SE_MobileNet1D_noLSTM:
Input: ECG beat (B, 1, 600)

→ Conv1D(1 → 32, kernel=7, stride=2, padding=3)
→ BatchNorm → ReLU

→ ResidualBlock(32):
    Conv1D(32, kernel=3, padding=1)
    → BN → ReLU
    → Conv1D(32, kernel=3, padding=1)
    → BN → +skip → ReLU

→ SEBlock(32):
    → GlobalAvgPool1D → (B, 32, 1)
    → Conv1D(32 → 4, kernel=1) → ReLU
    → Conv1D(4 → 32, kernel=1) → Sigmoid
    → scale input (B, 32, L)

→ Conv1D(32 → 64, kernel=5, stride=2, padding=2)
→ BatchNorm → ReLU

→ ResidualBlock(64):
    Conv1D(64, kernel=3, padding=1)
    → BN → ReLU
    → Conv1D(64, kernel=3, padding=1)
    → BN → +skip → ReLU

→ SEBlock(64):
    → GlobalAvgPool1D → (B, 64, 1)
    → Conv1D(64 → 8, kernel=1) → ReLU
    → Conv1D(8 → 64, kernel=1) → Sigmoid
    → scale input (B, 64, L)

→ GlobalAvgPool1D → shape: (B, 64)
→ Linear(64 → 32) → ReLU → Dropout(0.5)
→ Linear(32 → num_classes)

Output: Class logits (B, num_classes)


```

### 2. Beat-Type Classification – CNN Model (Pytorch)

Trained on beats centered on QRS complexes from:

* MIT-BIH Arrhythmia Database (mitdb)

* INCART 12-lead Arrhythmia Database (incartdb)

* SVDB 

* **Input**: 1D ECG window centered at QRS (540 samples of 360 hz signal)

* **Output**: Beat class (N, V, S , F, Q )

* **Deployment**: Used for inference after R-peak detection

#### Architecture Summary:

```
Conv1D(1 → 32, kernel=7, stride=2, padding=3)
→ BatchNorm → ReLU
→ ResidualBlock(32):
    Conv1D(32, kernel=3, padding=1)
    → BN → ReLU
    → Conv1D(32, kernel=3, padding=1)
    → BN → +skip → ReLU
→ SEBlock(32):
    → GlobalAvgPool
    → Conv1D(32 → 4) → ReLU → Conv1D(4 → 32) → Sigmoid
    → scale input
→ Conv1D(32 → 64, kernel=5, stride=2, padding=2)
→ BatchNorm → ReLU
→ ResidualBlock(64):
    Conv1D(64, kernel=3, padding=1)
    → BN → ReLU
    → Conv1D(64, kernel=3, padding=1)
    → BN → +skip → ReLU
→ SEBlock(64):
    → GlobalAvgPool
    → Conv1D(64 → 8) → ReLU → Conv1D(8 → 64) → Sigmoid
    → scale input
→ GlobalAvgPool1D
→ Flatten
→ Linear(64 → 32) → ReLU → Dropout(0.5)
→ Linear(32 → num_classes)

Output: Class logits (B, num_classes)


```

---


---

## 📊 Datasets Used

| Dataset        | Role                | Format | Notes                                    |
|----------------|---------------------|--------|------------------------------------------|
| CPSC 2018      | Rhythm classification | mat    | 1-lead, SNOMED codes                     |
| CPSC 2018 Extra| Rhythm classification | mat    | Additional records                       |
| PTB-XL         | Rhythm classification | mat    | With age and sex demographic data        |
| Georgia        | Rhythm classification | mat    | 12-lead                                   |
| MIT-BIH (mitdb)| Beat classification   | dat    | 360 Hz, QRS annotated                    |
| INCARTDB       | Beat classification   | dat    | 12-lead, annotated                       |
| SVDB           | Beat classification   | dat    | Supraventricular focus                  |

All signals were:
- Resampled to 500 Hz (rhythm) or 360 Hz (beat)
- Normalized and denoised using wavelet transform (`bior2.6`)
- Lead II extracted and used as input

---

## 📂 Signal Format

Saved recordings are stored in `.json` format:

```json
{
  "fs": 500,
  "lead": "II",
  "signal": [0.003, 0.002, 0.005 ],
  "start_time": "2025-06-01T12:01:00Z",
  "end_time": "2025-06-01T12:01:10Z"
}
```

---

## ⚠️ Medical Disclaimer

**This application is *not* a certified medical device.**

* It has not been evaluated by any regulatory authorities (FDA, CE, EMA).
* It is not intended for:

  * Diagnosing or monitoring medical conditions
  * Emergency or therapeutic use
* AI predictions are experimental and may be inaccurate or misleading.
* Signal quality may be affected by noise, motion, or hardware limitations.
* If you experience symptoms (e.g., chest pain, arrhythmia, dizziness), contact a qualified physician immediately.

> This app is intended **only for research, prototyping, and educational purposes**.

Use of this application is entirely at your own risk.

---

## 🧪 Use Cases

* ECG signal acquisition for biomedical prototyping
* BLE hardware integration (Arduino/ESP32)
* Testing real-time AI classification on mobile
* Study of rhythm and beat-type classification models

---

## 🔧 Tech Stack

* SwiftUI + CoreBluetooth (iOS frontend)
* PyTorch, Keras/TensorFlow (model training)
* ONNX → TensorFlow → CoreML / TFLite (deployment)
* WFDB, SciPy, PyWavelets (signal preprocessing)

---

## 📜 License

MIT License — For research and non-commercial use only.



---

## Authors

Marcin Sztukowski and Michal Naklicki
students of Jagiellonian University in Krakow
