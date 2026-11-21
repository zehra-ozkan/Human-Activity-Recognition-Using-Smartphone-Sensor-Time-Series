# Human Activity Recognition Using Smartphone Sensor Time Series

## Overview
This project investigates how the duration of smartphone sensor observations affects the accuracy, generalization to unseen users, and computational cost of Human Activity Recognition (HAR) models. Specifically, we evaluate how much sensor data is needed to reliably distinguish daily activities across previously unseen participants.

## Dataset
We use the **[WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)** (UCI Machine Learning Repository, DOI: [10.24432/C5HK59](https://doi.org/10.24432/C5HK59)), which contains raw three-axis accelerometer and gyroscope time-series measurements collected from **51 participants** performing **18 daily activities** at a sampling rate of approximately **20 Hz**. Each record includes a timestamp, an activity label, and three-axis sensor readings.

* **Selected Scope:** Both smartphone **accelerometer** `(ax, ay, az)` and **gyroscope** `(gx, gy, gz)` readings across the **Top 5 activities with the most samples** (selected via `src/activity_selection.py` across all 51 participants):
  1. `K` — **drinking** (487,585 total samples)
  2. `A` — **walking** (483,736 total samples)
  3. `M` — **kicking** (481,391 total samples)
  4. `P` — **dribbling** (475,061 total samples)
  5. `H` — **soup** (473,164 total samples)
* **Dataset Page:** [UCI Machine Learning Repository – WISDM Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
* **Dataset Documentation:** [WISDM-dataset-description.pdf](https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-dataset-description.pdf)

> **Note:** The smartphone accelerometer and gyroscope raw files (~456 MB) are included in the repository under `data/raw/`, so no separate download is needed. Generated files in `data/processed/` are committed to the repository (created by `src/preprocess.py`, which can be re-run to regenerate them).

## Setup
Requires **Python 3.12**.

### Windows (PowerShell)
```powershell
git clone https://github.com/zehra-ozkan/Human-Activity-Recognition-Using-Smartphone-Sensor-Time-Series.git
cd Human-Activity-Recognition-Using-Smartphone-Sensor-Time-Series

# Create and activate a Python 3.12 virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use Command Prompt: `.venv\Scripts\activate.bat`.)*

### macOS / Linux
```bash
git clone https://github.com/zehra-ozkan/Human-Activity-Recognition-Using-Smartphone-Sensor-Time-Series.git
cd Human-Activity-Recognition-Using-Smartphone-Sensor-Time-Series

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Methodology

### Preprocessing & Splitting (`src/activity_selection.py`, `src/preprocess.py`, `src/features.py`)
* **Activity Selection (`src/activity_selection.py`):** Scans all raw smartphone accelerometer and gyroscope files, ranks all 18 activities by total sample count, and saves the top-5 activities (`K: drinking`, `A: walking`, `M: kicking`, `P: dribbling`, `H: soup`) to `data/processed/selected_activities.json`.
* **Cleaning, Ordering & 20 Hz Sensor Alignment (`src/preprocess.py`):** Raw accelerometer and gyroscope recordings are cleaned (trailing `;` stripped, malformed/NaN rows dropped), sorted chronologically by nanosecond timestamp, and deduplicated. Recordings are split at any timestamp gap $> 0.5\text{ s}$ into contiguous bouts and linearly interpolated onto a shared **20 Hz (50 ms)** time grid to form 6-channel time series `(ax, ay, az, gx, gy, gz)` (255 contiguous bouts; 916,901 aligned time steps).
* **Participant-Disjoint Split (`splits.json`):** The 51 participants are split strictly by participant ID into **36 training users (~70%)**, **8 validation users (~15%)**, and **7 test users (~15%)** (`seed=42`). Zero windows from validation or test users appear in training, preventing data leakage and evaluating generalization on unseen users.
* **Multi-Duration Windowing:** Aligned 6-channel streams are segmented into non-overlapping fixed-duration windows of **1, 2, 5, and 10 seconds** (20, 40, 100, and 200 time steps at 20 Hz), yielding **45,665** (1s), **22,715** (2s), **8,945** (5s), and **4,355** (10s) windows.
* **Dual Representations & Leak-Free Normalization (`src/features.py`):**
  * **Traditional ML Features (`data/processed/ml/features_{1,2,5,10}s.npz` & `.csv`):** Extracts **94 statistical features** per window — 11 time- and frequency-domain metrics (`mean`, `std`, `min`, `max`, `median`, `iqr`, `rms`, `mad`, `zcr`, `spec_energy`, `dom_freq_mag`) across 8 signals (`ax, ay, az, gx, gy, gz, accel_mag, gyro_mag`) plus 6 pairwise axis correlations.
  * **Deep Learning Sequences (`data/processed/dl/sequences_{1,2,5,10}s.npz`):** Preserves 6-channel sensor sequences of shape `(N, T, 6)`.
  * **Why Scalers Are Fit Strictly on the 36 Training Participants:** Validation and test participants represent future, unseen users whose physical movement distributions are unknown at training time. Both the 94-feature Z-score standardizer and the 6-channel sequence Z-score normalizer compute $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ **only on the 36 training participants** and apply those parameters unchanged to transform train, validation, and test sets ($z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$).

### Running the Preprocessing Pipeline
```bash
# 1. Select the top-5 activities with the most samples
python src/activity_selection.py

# 2. Clean raw data, align accel+gyro at 20 Hz, split participants, window (1/2/5/10s), and extract ML & DL representations
python src/preprocess.py
```

### Models
* **Traditional Machine Learning Baselines:** Random Forest, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP).
* **Deep Learning Time-Series Models:** Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks.

### Planned Experiments
* Compare **accuracy** and **macro-averaged F1-score** across 1-, 2-, 5-, and 10-second observation windows.
* Evaluate generalization performance on previously unseen participants.
* Analyze per-activity recognition performance to identify hard-to-distinguish activities and test whether longer windows improve recognition.
* Investigate whether class balancing improves performance on underrepresented activities.

## Repository Structure
```text
.
├── data/
│   ├── raw/                 # Raw WISDM smartphone files (committed)
│   │   ├── accel/           # Smartphone accelerometer, one file per subject
│   │   ├── gyro/            # Smartphone gyroscope, one file per subject
│   │   └── activity_key.txt # Activity letter codes (A–S)
│   └── processed/           # Windowed features and normalized sequences (gitignored)
│       ├── selected_activities.json
│       ├── preprocessing_summary.json
│       ├── raw_windows/     # Unnormalized 6-channel windows: windows_{1,2,5,10}s.npz
│       ├── ml/              # 94 statistical features: features_{1,2,5,10}s.{npz,csv}
│       └── dl/              # Z-score normalized sequences: sequences_{1,2,5,10}s.npz
├── notebooks/               # Exploratory data analysis (EDA) and visualization
├── src/
│   ├── activity_selection.py # Ranks activities by sample count and selects top-5
│   ├── eda.py               # Dataset statistics and class balance plots
│   ├── preprocess.py        # Cleaning, timestamp ordering, 20 Hz alignment, windowing, and participant split
│   ├── features.py          # 94 statistical features and train-only Z-score normalization
│   ├── train_ml.py          # Random Forest, SVM, and MLP training & evaluation
│   └── train_dl.py          # LSTM and GRU training & evaluation
├── results/                 # Saved metrics, confusion matrices, and runtime logs
├── splits.json              # Shared participant-disjoint train/val/test split
├── requirements.txt
└── README.md
