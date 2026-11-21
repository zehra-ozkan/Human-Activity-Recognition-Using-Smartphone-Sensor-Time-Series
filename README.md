# Human Activity Recognition Using Smartphone Sensor Time Series

## Overview
This project investigates how the duration of smartphone sensor observations affects the accuracy, generalization to unseen users, and computational cost of Human Activity Recognition (HAR) models. Specifically, we evaluate how much sensor data is needed to reliably distinguish daily activities across previously unseen participants.

## Dataset
We use the **[WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)** (UCI Machine Learning Repository, DOI: [10.24432/C5HK59](https://doi.org/10.24432/C5HK59)), which contains raw three-axis accelerometer and gyroscope time-series measurements collected from **51 participants** performing **18 daily activities** at a sampling rate of approximately **20 Hz**. Each record includes a timestamp, an activity label, and three-axis sensor readings.

* **Initial Scope:** Smartphone accelerometer readings across a selected subset of activities—`walking`, `jogging`, `stairs`, `sitting`, `standing`, and `typing`.
* **Dataset Page:** [UCI Machine Learning Repository – WISDM Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
* **Dataset Documentation:** [WISDM-dataset-description.pdf](https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-dataset-description.pdf)

> **Note:** The smartphone accelerometer and gyroscope raw files (~456 MB) are included in the repository under `data/raw/`, so no separate download is needed. Generated files in `data/processed/` are gitignored and created by `src/preprocess.py`.

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

### Preprocessing & Splitting
* **Cleaning & Ordering:** Raw sensor recordings are cleaned and chronologically organized by timestamp.
* **Windowing:** Time series are segmented into fixed-duration windows of **1, 2, 5, and 10 seconds** with assigned activity labels.
* **Representations:** Statistical features are extracted from each window for traditional machine learning models, while normalized sensor sequences are prepared for deep learning models.
* **Participant-Disjoint Split:** Data is split strictly by participant ID rather than random windows to prevent data leakage and evaluate generalization on previously unseen users.

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
├── notebooks/               # Exploratory data analysis (EDA) and visualization
├── src/
│   ├── eda.py               # Dataset statistics and class balance plots
│   ├── preprocess.py        # Cleaning, timestamp ordering, windowing, and participant split
│   ├── features.py          # Statistical feature extraction and sequence normalization
│   ├── train_ml.py          # Random Forest, SVM, and MLP training & evaluation
│   └── train_dl.py          # LSTM and GRU training & evaluation
├── results/                 # Saved metrics, confusion matrices, and runtime logs
├── splits.json              # Shared participant-disjoint train/val/test split
├── requirements.txt
└── README.md
