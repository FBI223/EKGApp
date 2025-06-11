



# 📱 ECGApp – Mobile ECG Viewer & Classifier

**ECGApp** is a cross-platform mobile application (iOS ) for real-time ECG signal acquisition, processing, and classification using deep learning models.

---


## 📚 Table of Contents
- [Key Features](#-key-features)
- [Used Deep Learning Models](#-used-deep-learning-models)
- [Datasets Used](#-datasets-used)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Signal Format](#-signal-format)
- [Import & Interoperability](#-import--interoperability)
- [Use Cases](#-use-cases)
- [Tech Stack](#-tech-stack)
- [Medical Disclaimer](#-medical-disclaimer)


---

## 🧠 Key Features

* 🔌 Connect to external BLE ECG sensors (e.g., ESP32, Arduino)
* 📈 Real-time signal visualization (zoom, pan, rescale)
* ⏺ Record ECG to local `.json` and `.dat` and `.hea` format
* 🧠 AI-based classification:

  * Heart **rhythm classification** (e.g. NSR, AF, PVC)
  * **Beat-type classification** near QRS centers (e.g. N, V, S)
   * **Waveform segmentation** per sample into `P`, `QRS`, `T` waves for morphological analysis
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
→ Conv1D(1 → 16, kernel=7, stride=2, padding=3)
  → BatchNorm1d(16)
  → SiLU

→ DepthwiseConv1D(16 → 16, kernel=5, stride=2, padding=2, groups=16)
→ PointwiseConv1D(16 → 32, kernel=1)
  → BatchNorm1d(32)
  → SiLU
→ SEBlock(32):
     → AdaptiveAvgPool1d(1)
     → Linear(32 → 4) → ReLU
     → Linear(4 → 32) → Sigmoid
     → Multiply input × scale
→ Dropout(0.1)

→ DepthwiseConv1D(32 → 32, kernel=5, stride=2, padding=2, groups=32)
→ PointwiseConv1D(32 → 64, kernel=1)
  → BatchNorm1d(64)
  → SiLU
→ SEBlock(64):
     → AdaptiveAvgPool1d(1)
     → Linear(64 → 8) → ReLU
     → Linear(8 → 64) → Sigmoid
     → Multiply input × scale
→ Dropout(0.1)

→ DepthwiseConv1D(64 → 64, kernel=5, stride=2, padding=2, groups=64)
→ PointwiseConv1D(64 → 128, kernel=1)
  → BatchNorm1d(128)
  → SiLU
→ SEBlock(128):
     → AdaptiveAvgPool1d(1)
     → Linear(128 → 16) → ReLU
     → Linear(16 → 128) → Sigmoid
     → Multiply input × scale
→ Dropout(0.1)

→ DepthwiseConv1D(128 → 128, kernel=5, stride=1, padding=2, groups=128)
→ PointwiseConv1D(128 → 128, kernel=1)
  → BatchNorm1d(128)
  → SiLU
→ SEBlock(128):
     → AdaptiveAvgPool1d(1)
     → Linear(128 → 16) → ReLU
     → Linear(16 → 128) → Sigmoid
     → Multiply input × scale
→ Dropout(0.1)


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
ECGClassifier:
Input: (B, 1, T) (np. 540 próbek @ 360 Hz)



FEATURE EXTRACTION
→ Conv1D(1 → 32, kernel=7, stride=2, padding=3)
→ BatchNorm1d(32) → ReLU

→ ResidualBlock(32):
  → Conv1D(32 → 32, kernel=3, padding=1)
  → BatchNorm1d(32) → ReLU
  → Conv1D(32 → 32, kernel=3, padding=1)
  → BatchNorm1d(32)
  → Skip connection + ReLU

→ SEBlock(32):
  → AdaptiveAvgPool1d(1)
  → Conv1D(32 → 4, kernel=1) → ReLU
  → Conv1D(4 → 32, kernel=1) → Sigmoid
  → Multiply (channel-wise scaling)

→ Conv1D(32 → 64, kernel=5, stride=2, padding=2)
→ BatchNorm1d(64) → ReLU

→ ResidualBlock(64):
  → Conv1D(64 → 64, kernel=3, padding=1)
  → BatchNorm1d(64) → ReLU
  → Conv1D(64 → 64, kernel=3, padding=1)
  → BatchNorm1d(64)
  → Skip connection + ReLU

→ SEBlock(64):
  → AdaptiveAvgPool1d(1)
  → Conv1D(64 → 8, kernel=1) → ReLU
  → Conv1D(8 → 64, kernel=1) → Sigmoid
  → Multiply (channel-wise scaling)




CLASSIFICATION HEAD
→ AdaptiveAvgPool1d(1)
→ Flatten: (B, 64)

→ Linear(64 → 32)
→ ReLU
→ Dropout(0.5)

→ Linear(32 → num_classes)

Output:
→ Shape: (B, num_classes) – logits of classes.

```

---

---

### 3. Waveform Segmentation – UNet1D (PyTorch)
Lightweight 1D U-Net model for per-sample waveform classification trained on LUDB dataset.

* **Input**: 10-second 1-lead ECG segment (2000 samples of 500 Hz signal, lead II)
* **Output**: Per-sample waveform class (none, P, QRS, T) using softmax over 4 classes
* **Deployment**: Converted to CoreML (UnetModel.mlpackage) for real-time waveform segmentation
* **Datasets**: LUDB (Lobachevsky University Database), 200 manually annotated 12-lead ECGs



#### Architecture Summary:

```
UNet1D:
Input: (B, 1, 2000)


ENCODER / DOWNSAMPLING

→ Block 1:
  → Conv1D(1 → 4, kernel=9, padding=4)
  → BatchNorm1d(4) → ReLU
  → Conv1D(4 → 4, kernel=9, padding=4)
  → BatchNorm1d(4) → ReLU
  → MaxPool1D(kernel=2)             # ↓ T/2

→ Block 2:
  → Conv1D(4 → 8, kernel=9, padding=4)
  → BatchNorm1d(8) → ReLU
  → Conv1D(8 → 8, kernel=9, padding=4)
  → BatchNorm1d(8) → ReLU
  → MaxPool1D(kernel=2)             # ↓ T/4

→ Block 3:
  → Conv1D(8 → 16, kernel=9, padding=4)
  → BatchNorm1d(16) → ReLU
  → Conv1D(16 → 16, kernel=9, padding=4)
  → BatchNorm1d(16) → ReLU
  → MaxPool1D(kernel=2)             # ↓ T/8

→ Block 4:
  → Conv1D(16 → 32, kernel=9, padding=4)
  → BatchNorm1d(32) → ReLU
  → Conv1D(32 → 32, kernel=9, padding=4)
  → BatchNorm1d(32) → ReLU
  → MaxPool1D(kernel=2)             # ↓ T/16




BOTTLENECK

→ Conv1D(32 → 64, kernel=9, padding=4)
→ BatchNorm1d(64) → ReLU
→ Conv1D(64 → 64, kernel=9, padding=4)
→ BatchNorm1d(64) → ReLU




DECODER / UPSAMPLING

→ TransposedConv1D(64 → 32, kernel=8, stride=2, padding=3)
→ Pad to match enc4 → Concat([up, enc4]) → (64 channels)
→ Conv1D(64 → 32) → BN → ReLU → Conv1D → BN → ReLU

→ TransposedConv1D(32 → 16, kernel=8, stride=2, padding=3)
→ Pad to match enc3 → Concat([up, enc3]) → (32 channels)
→ Conv1D(32 → 16) → BN → ReLU → Conv1D → BN → ReLU

→ TransposedConv1D(16 → 8, kernel=8, stride=2, padding=3)
→ Pad to match enc2 → Concat([up, enc2]) → (16 channels)
→ Conv1D(16 → 8) → BN → ReLU → Conv1D → BN → ReLU

→ TransposedConv1D(8 → 4, kernel=8, stride=2, padding=3)
→ Pad to match enc1 → Concat([up, enc1]) → (8 channels)
→ Conv1D(8 → 4) → BN → ReLU → Conv1D → BN → ReLU



OUTPUT

→ Final Conv1D(4 → num_classes, kernel=1)  # Pointwise convolution
→ Output shape: (B, num_classes, T)
→ Permute to (B, T, num_classes) for per-sample classification




```


---


---

## 📊 Datasets Used

| Dataset         | Role                  | Format | Notes                             |
|-----------------|-----------------------|--------|-----------------------------------|
| CPSC 2018       | Rhythm classification | mat    | 1-lead, SNOMED codes              |
| CPSC 2018 Extra | Rhythm classification | mat    | Additional records                |
| PTB-XL          | Rhythm classification | mat    | With age and sex demographic data |
| Georgia         | Rhythm classification | mat    | 12-lead                           |
| MIT-BIH (mitdb) | Beat classification   | dat    | 360 Hz, QRS annotated             |
| INCARTDB        | Beat classification   | dat    | 12-lead, annotated                |
| SVDB            | Beat classification   | dat    | Supraventricular focus            |
| LUDB            | Wave classification   | dat    | Waves focus                       |


All signals were:
- Resampled to 500 Hz (rhythm and wave) or 360 Hz (beat)
- Normalized and denoised using wavelet transform (`bior2.6`)
- Lead II extracted and used as input

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


## 🔍 Evaluation & Metrics

Each model is evaluated with task-specific metrics:

### Rhythm Classification
- ✅ Accuracy, macro/weighted F1-score  
- ✅ Per-class precision and recall  
- ✅ Confusion matrix (10-class)  
- ✅ NSR fallback threshold:  
  - `softmax max < 0.4` → fallback to NSR (`59118001`)

### Beat-Type Classification
- ✅ Per-beat accuracy (N, V, S, F, Q)  
- ✅ Inference on 540-sample QRS-centered segments  
- ✅ Real-time aggregation of beat-type predictions

### Waveform Segmentation
- ✅ Sample-wise F1-score and accuracy  
- ✅ Onset/offset match with ±150 sample tolerance  
- ✅ Segment-wise precision, recall, and F1-score  
- ✅ Overlay visualization: predicted vs. true waveforms

---



## 📂 Signal Format

### Saved recordings are stored in `.json` format:

```json
{
  "fs": 500,
  "leads": ["II"],
  "signals": [[0.003, 0.002, 0.005]],
  "start_time": "2025-06-01T12:01:00Z",
  "end_time": "2025-06-01T12:01:10Z"
}
```
* **fs**: Sampling frequency (Hz)
* **leads**: Array of lead names (e.g., ["II"])
+ **signals**: List of signal arrays (1 per lead)
* **start_time**, end_time: ISO 8601 timestamps

---
### Saved recordings are stored in `.hea` format:
```
record123 1 500 5000
record123.dat 16 1000 16 0 0 200 0 II
# age: 26
# sex: M
# duration: 10 seconds
# start_time: 2025-06-01T12:01:00Z
# end_time: 2025-06-01T12:01:10Z
# Recorded via ECG mobile app

```
* **Line** 1: <record_name> <n_signals> <fs> <n_samples>
* **Line** 2: <file> <format> <gain> <bit_res> <adc_zero> <init_val> <adc_resolution> 0 <lead>
* **Comments** (#) include age, sex, start/end time, duration
---
### Saved recordings are stored in `.dat` format:

Signal samples saved as Int16, scaled by gain
Written in little-endian format
Lead order matches .hea

---

## 📥 Import & Interoperability

The app supports importing ECG files in multiple formats:

- `.dat` + `.hea` (WFDB-compliant):
  - e.g., MITDB, INCARTDB, LUDB
- `.json` format exported by the ECGApp

All imported signals are parsed and normalized, including:
- Sampling frequency (`fs`)
- Number of samples
- Gain, resolution, zero offset
- Lead name (e.g., II)

Supported encoding:
- 🧩 16-bit integer (`format 16`)
- ✅ Single-lead inputs (Lead II preferred)

---


## ☁️ iCloud & File Sharing

ECGApp fully integrates with the iOS Files system, enabling secure cloud-based data access:

- 📤 **Export ECG recordings** to iCloud Drive for backup or manual inspection  
- 📥 **Import** `.dat` / `.hea` / `.json` files from iCloud, USB, or AirDrop  
- 🔁 Seamlessly **transfer data** between iPhone, iPad, or Mac  
- 📁 Files are available under **Files → ECGApp → On My iPhone**

This makes it easy to analyze, review, or back up ECG sessions securely.

---

## 🛡️ Data Privacy

All ECG data is processed and stored **locally on the user’s device**.  
No recordings or personal information are sent to external servers.

- 📱 All AI inference and signal analysis are performed **on-device**
- ☁️ If the user chooses to share recordings via iCloud, AirDrop, or Files, **the transfer is secure and user-controlled**
- 🛑 The app does **not collect, upload, or transmit** any data automatically
- ✅ No third-party analytics, tracking, or background syncing

Users have full control over their data:

- 📂 Access and manage saved recordings (`.json`, `.dat`, `.hea`)
- 🗑 Manually delete or export any file
- 🔒 Bluetooth connections are limited to **approved UUIDs only**

> Your data remains yours — private, secure, and under your control.


---

## 📦 Future Work (Planned)

- 💡 Multi-lead ECG support (e.g., V1, V5, aVR)
- 🧪 Personalized on-device model fine-tuning
- 🩺 AI-based tagging of symptoms and abnormalities
- 📈 Long-term trend tracking and visualization


---

## 📜 License

MIT License — For research and non-commercial use only.



---

## Authors

Marcin Sztukowski 
student of Jagiellonian University in Krakow
