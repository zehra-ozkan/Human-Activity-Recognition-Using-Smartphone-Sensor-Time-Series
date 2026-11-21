"""Statistical feature extraction and sequence normalization for HAR windows.

Provides two representations from each 6-channel sensor window
``(ax, ay, az, gx, gy, gz)`` sampled at 20 Hz:
1. **Statistical Features** (for Random Forest, SVM, MLP):
   - Per-channel time & frequency domain statistics (11 features x 6 channels = 66)
   - Vector magnitude statistics for accelerometer ``||a||`` and gyroscope ``||g||``
     (11 features x 2 magnitudes = 22)
   - Pairwise axis correlations ``corr(x,y), corr(x,z), corr(y,z)`` for accel & gyro (6)
   - Total: **94 features** per window, standardized with a Z-score scaler fit
     strictly on the 36 training participants to prevent data leakage.
2. **Normalized Sensor Sequences** (for LSTM, GRU):
   - Per-channel Z-score normalization fit strictly on the 36 training participants'
     time steps and applied to train, validation, and test sequences.

Why normalization/standardization is fit ONLY on the 36 training participants:
    In real-world deployment, validation (8 users) and test (7 users) participants
    represent future, unseen users whose sensor distributions (movement intensity,
    walking speed, phone orientation in pocket) are unknown at training time.
    Computing mean (mu) and standard deviation (sigma) across all 51 participants
    before splitting—or normalizing test users with their own future timestamps—would
    cause data leakage. Therefore, mu_train and sigma_train are computed strictly on
    X_train (the 36 training users) and applied unchanged to transform train, val,
    and test sets: z = (x - mu_train) / sigma_train.

Usage examples (run from the project root directory):
    # 1. Normally invoked automatically by src/preprocess.py, or standalone after windowing:
    python src/features.py

    # 2. Re-extract features and normalized sequences for specific window durations:
    python src/features.py --windows 1 2 5 10

    # 3. Specify a custom processed data directory:
    python src/features.py --processed-dir data/processed

    # 4. View all command-line options:
    python src/features.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHANNEL_NAMES: Tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")
MAGNITUDE_NAMES: Tuple[str, ...] = ("accel_mag", "gyro_mag")
STAT_NAMES: Tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "rms",
    "mad",
    "zcr",
    "spec_energy",
    "dom_freq_mag",
)


def get_feature_names() -> List[str]:
    """Return ordered list of all 94 statistical feature names."""
    names: List[str] = []
    for ch in (*CHANNEL_NAMES, *MAGNITUDE_NAMES):
        for st in STAT_NAMES:
            names.append(f"{ch}_{st}")
    names.extend(
        [
            "accel_corr_xy",
            "accel_corr_xz",
            "accel_corr_yz",
            "gyro_corr_xy",
            "gyro_corr_xz",
            "gyro_corr_yz",
        ]
    )
    return names


def _safe_corrcoef_pairs(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute vectorized pairwise Pearson correlation (xy, xz, yz) across windows.

    Args:
        x, y, z: Arrays of shape ``(N, T)``.

    Returns:
        Array of shape ``(N, 3)`` with correlations in ``[-1, 1]`` (0 if constant signal).
    """
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    zm = z - z.mean(axis=1, keepdims=True)

    sx = np.sqrt(np.sum(xm * xm, axis=1))
    sy = np.sqrt(np.sum(ym * ym, axis=1))
    sz = np.sqrt(np.sum(zm * zm, axis=1))

    eps = 1e-12
    c_xy = np.sum(xm * ym, axis=1) / np.maximum(sx * sy, eps)
    c_xz = np.sum(xm * zm, axis=1) / np.maximum(sx * sz, eps)
    c_yz = np.sum(ym * zm, axis=1) / np.maximum(sy * sz, eps)

    c_xy = np.where((sx > eps) & (sy > eps), np.clip(c_xy, -1.0, 1.0), 0.0)
    c_xz = np.where((sx > eps) & (sz > eps), np.clip(c_xz, -1.0, 1.0), 0.0)
    c_yz = np.where((sy > eps) & (sz > eps), np.clip(c_yz, -1.0, 1.0), 0.0)
    return np.stack([c_xy, c_xz, c_yz], axis=1)


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """Extract 94 vectorized statistical features from raw windows.

    Args:
        X: Windowed sensor array of shape ``(N, T, 6)`` with channels
           ``(ax, ay, az, gx, gy, gz)``.

    Returns:
        Feature matrix of shape ``(N, 94)`` as ``float32``.
    """
    if X.ndim != 3 or X.shape[2] != 6:
        raise ValueError(f"Expected X of shape (N, T, 6), got {X.shape}")

    X64 = X.astype(np.float64, copy=False)
    accel_mag = np.linalg.norm(X64[:, :, 0:3], axis=2, keepdims=True)
    gyro_mag = np.linalg.norm(X64[:, :, 3:6], axis=2, keepdims=True)
    signals = np.concatenate([X64, accel_mag, gyro_mag], axis=2)  # (N, T, 8)

    N, T, C = signals.shape
    mean = np.mean(signals, axis=1)
    std = np.std(signals, axis=1)
    vmin = np.min(signals, axis=1)
    vmax = np.max(signals, axis=1)
    q25, median, q75 = np.percentile(signals, [25.0, 50.0, 75.0], axis=1)
    iqr = q75 - q25
    rms = np.sqrt(np.mean(signals * signals, axis=1))

    centered = signals - mean[:, None, :]
    mad = np.mean(np.abs(centered), axis=1)

    # Zero-crossing rate of mean-centered signal
    signs = np.sign(centered)
    zcr = np.mean(signs[:, 1:, :] * signs[:, :-1, :] < 0, axis=1)

    # Frequency-domain features via real FFT (excluding DC bin 0)
    fft_mag = np.abs(np.fft.rfft(centered, axis=1))
    if fft_mag.shape[1] > 1:
        ac_mag = fft_mag[:, 1:, :]
        spec_energy = np.mean(ac_mag * ac_mag, axis=1)
        dom_freq_mag = np.max(ac_mag, axis=1)
    else:
        spec_energy = np.zeros((N, C), dtype=np.float64)
        dom_freq_mag = np.zeros((N, C), dtype=np.float64)

    # Stack 11 stats per signal -> (N, 8, 11) -> flatten to (N, 88)
    per_signal_stats = np.stack(
        [
            mean,
            std,
            vmin,
            vmax,
            median,
            iqr,
            rms,
            mad,
            zcr,
            spec_energy,
            dom_freq_mag,
        ],
        axis=2,
    ).reshape(N, C * len(STAT_NAMES))

    accel_corr = _safe_corrcoef_pairs(X64[:, :, 0], X64[:, :, 1], X64[:, :, 2])
    gyro_corr = _safe_corrcoef_pairs(X64[:, :, 3], X64[:, :, 4], X64[:, :, 5])

    features = np.concatenate([per_signal_stats, accel_corr, gyro_corr], axis=1)
    return features.astype(np.float32)


def fit_sequence_scaler(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-channel mean and std across all time steps of training windows only."""
    flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_sequences(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply training channel mean/std Z-score normalization to sequences."""
    return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def fit_feature_scaler(F_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-feature mean and std on training feature matrix only."""
    f64 = F_train.astype(np.float64)
    mean = f64.mean(axis=0)
    std = f64.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_features(
    F: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Standardize feature matrix using training mean and std."""
    return ((F - mean[None, :]) / std[None, :]).astype(np.float32)


def build_representations_for_window(
    window_sec: int,
    splits_data: Dict[str, np.ndarray],
    idx_to_code: Dict[int, str],
    idx_to_activity: Dict[int, str],
    processed_dir: Path = PROCESSED_DIR,
) -> Dict:
    """Save normalized sequences (DL) and standardized statistical features (ML).

    Args:
        window_sec: Window duration in seconds (e.g. 1, 2, 5, 10).
        splits_data: Dict containing ``X_train, y_train, subj_train, X_val, y_val,
            subj_val, X_test, y_test, subj_test``.
        idx_to_code: Mapping from integer label ``0..K-1`` to activity code (e.g. ``'K'``).
        idx_to_activity: Mapping from integer label ``0..K-1`` to activity name.
        processed_dir: Target output directory (`data/processed`).
    """
    dl_dir = processed_dir / "dl"
    ml_dir = processed_dir / "ml"
    dl_dir.mkdir(parents=True, exist_ok=True)
    ml_dir.mkdir(parents=True, exist_ok=True)

    # 1. Deep Learning normalized sequences (fit strictly on X_train)
    seq_mean, seq_std = fit_sequence_scaler(splits_data["X_train"])
    X_train_norm = transform_sequences(splits_data["X_train"], seq_mean, seq_std)
    X_val_norm = transform_sequences(splits_data["X_val"], seq_mean, seq_std)
    X_test_norm = transform_sequences(splits_data["X_test"], seq_mean, seq_std)

    dl_path = dl_dir / f"sequences_{window_sec}s.npz"
    np.savez_compressed(
        dl_path,
        X_train=X_train_norm,
        y_train=splits_data["y_train"],
        subj_train=splits_data["subj_train"],
        X_val=X_val_norm,
        y_val=splits_data["y_val"],
        subj_val=splits_data["subj_val"],
        X_test=X_test_norm,
        y_test=splits_data["y_test"],
        subj_test=splits_data["subj_test"],
        channel_names=np.array(CHANNEL_NAMES),
        scaler_mean=seq_mean,
        scaler_std=seq_std,
    )

    # 2. Traditional ML statistical features (extracted from unnormalized windows,
    #    then standardized using training participants only)
    feature_names = get_feature_names()
    F_train_raw = extract_statistical_features(splits_data["X_train"])
    F_val_raw = extract_statistical_features(splits_data["X_val"])
    F_test_raw = extract_statistical_features(splits_data["X_test"])

    feat_mean, feat_std = fit_feature_scaler(F_train_raw)
    F_train = transform_features(F_train_raw, feat_mean, feat_std)
    F_val = transform_features(F_val_raw, feat_mean, feat_std)
    F_test = transform_features(F_test_raw, feat_mean, feat_std)

    ml_npz_path = ml_dir / f"features_{window_sec}s.npz"
    np.savez_compressed(
        ml_npz_path,
        X_train=F_train,
        y_train=splits_data["y_train"],
        subj_train=splits_data["subj_train"],
        X_val=F_val,
        y_val=splits_data["y_val"],
        subj_val=splits_data["subj_val"],
        X_test=F_test,
        y_test=splits_data["y_test"],
        subj_test=splits_data["subj_test"],
        X_train_unscaled=F_train_raw,
        X_val_unscaled=F_val_raw,
        X_test_unscaled=F_test_raw,
        feature_names=np.array(feature_names),
        scaler_mean=feat_mean,
        scaler_std=feat_std,
    )

    # Also save an easy-to-inspect CSV of standardized features with metadata
    dfs = []
    for split_name, F_split, y_split, s_split in [
        ("train", F_train, splits_data["y_train"], splits_data["subj_train"]),
        ("val", F_val, splits_data["y_val"], splits_data["subj_val"]),
        ("test", F_test, splits_data["y_test"], splits_data["subj_test"]),
    ]:
        df = pd.DataFrame(F_split, columns=feature_names)
        df.insert(0, "split", split_name)
        df.insert(1, "participant", s_split)
        df.insert(2, "label", y_split)
        df.insert(3, "activity_code", [idx_to_code[int(i)] for i in y_split])
        df.insert(4, "activity_name", [idx_to_activity[int(i)] for i in y_split])
        dfs.append(df)

    ml_csv_path = ml_dir / f"features_{window_sec}s.csv"
    pd.concat(dfs, ignore_index=True).to_csv(ml_csv_path, index=False)

    return {
        "window_sec": window_sec,
        "num_features": len(feature_names),
        "dl_path": str(dl_path.relative_to(PROJECT_ROOT)),
        "ml_npz_path": str(ml_npz_path.relative_to(PROJECT_ROOT)),
        "ml_csv_path": str(ml_csv_path.relative_to(PROJECT_ROOT)),
        "sequence_shape_train": list(X_train_norm.shape),
        "sequence_shape_val": list(X_val_norm.shape),
        "sequence_shape_test": list(X_test_norm.shape),
        "feature_shape_train": list(F_train.shape),
        "feature_shape_val": list(F_val.shape),
        "feature_shape_test": list(F_test.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-generate statistical features and normalized DL sequences from saved raw windows."
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Window durations in seconds (default: 1 2 5 10)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing preprocessed raw windows (default: data/processed)",
    )
    args = parser.parse_args()

    meta_path = args.processed_dir / "preprocessing_summary.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run 'python src/preprocess.py' first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    idx_to_code = {int(k): v for k, v in meta["label_to_code"].items()}
    idx_to_activity = {int(k): v for k, v in meta["label_to_activity"].items()}

    for w in args.windows:
        raw_win_path = args.processed_dir / "raw_windows" / f"windows_{w}s.npz"
        with np.load(raw_win_path) as data:
            splits_data = {k: data[k] for k in data.files}
        info = build_representations_for_window(
            window_sec=w,
            splits_data=splits_data,
            idx_to_code=idx_to_code,
            idx_to_activity=idx_to_activity,
            processed_dir=args.processed_dir,
        )
        print(
            f"[{w:>2}s] DL train={tuple(info['sequence_shape_train'])} | "
            f"ML train={tuple(info['feature_shape_train'])} -> {info['ml_npz_path']}"
        )


if __name__ == "__main__":
    main()
