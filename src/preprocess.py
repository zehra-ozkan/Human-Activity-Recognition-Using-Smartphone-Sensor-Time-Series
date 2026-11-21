"""End-to-end preprocessing pipeline for WISDM Smartphone Sensor Time Series.

Performs all steps required by the project methodology:
1. **Activity Filtering**: Loads the selected top-5 activities from
   ``data/processed/selected_activities.json`` (or runs ``activity_selection.py``
   if not yet created).
2. **Cleaning & Timestamp Ordering**: Parses raw smartphone accelerometer and
   gyroscope files, strips trailing semicolons, drops malformed/NaN rows, sorts
   chronologically by nanosecond timestamp, and deduplicates identical timestamps.
3. **Contiguous Bout Detection & 20 Hz Alignment**: Splits recordings whenever
   a sensor gap exceeds ``--max-gap-sec`` (default: 0.5 s) and interpolates
   accelerometer ``(ax, ay, az)`` and gyroscope ``(gx, gy, gz)`` onto a shared
   20 Hz (50 ms) time grid over their overlapping time span.
4. **Participant-Disjoint Splitting**: Splits the 51 participants strictly by
   participant ID into train (70%, 36 users), validation (15%, 8 users), and
   test (15%, 7 users) sets and saves ``splits.json`` at the project root to
   prevent data leakage and evaluate generalization on unseen users.
5. **Multi-Duration Windowing (1s, 2s, 5s, 10s)**: Segments aligned 6-channel
   recordings into fixed-duration windows of 1, 2, 5, and 10 seconds (20, 40,
   100, and 200 samples at 20 Hz) with configurable overlap (default: 0.0 for
   non-overlapping windows).
6. **Dual Representations & Leak-Free Scaling** (via ``src/features.py``):
   - Extracts **94 statistical features** per window and standardizes them
     using mean/std fit strictly on the 36 training participants for traditional
     ML models (RF, SVM, MLP).
   - Normalizes **6-channel sensor sequences** per channel using mean/std fit
     strictly on the 36 training participants for deep learning models (LSTM, GRU).
   - Why train-only scaling: Validation and test participants represent future,
     unseen users; fitting scalers on their data would leak test distribution
     statistics into training.

Usage examples (run from the project root directory):
    # 1. Default full pipeline (1s, 2s, 5s, 10s windows, top-5 activities, 70/15/15 participant split)
    python src/preprocess.py

    # 2. Customize window sizes or add 50% sliding-window overlap
    python src/preprocess.py --windows 1 2 5 10 --overlap 0.5

    # 3. Change random seed or train/val/test split ratios
    python src/preprocess.py --seed 42 --train-ratio 0.70 --val-ratio 0.15

    # 4. View all command-line options
    python src/preprocess.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_selection import (  # noqa: E402
    DEFAULT_OUTPUT_PATH as SELECTED_ACTIVITIES_PATH,
    select_top_activities,
)
from features import (  # noqa: E402
    CHANNEL_NAMES,
    build_representations_for_window,
    get_feature_names,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_PATH = PROJECT_ROOT / "splits.json"

SAMPLING_RATE_HZ = 20
STEP_NS = int(1_000_000_000 // SAMPLING_RATE_HZ)  # 50,000,000 ns (50 ms)


def parse_raw_sensor_file(
    file_path: Path, allowed_codes: Set[str]
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    """Read, clean, sort, and deduplicate a single participant's raw sensor file.

    Args:
        file_path: Path to ``data_<id>_accel_phone.txt`` or ``data_<id>_gyro_phone.txt``.
        allowed_codes: Set of selected activity letter codes (e.g. ``{'K','A','M','P','H'}``).

    Returns:
        - Dict mapping activity code -> ``(timestamps_ns, xyz_values)`` sorted strictly
          in ascending timestamp order with duplicates averaged.
        - Dict of cleaning statistics (total lines, kept rows, malformed rows, duplicate timestamps).
    """
    raw_ts: Dict[str, List[int]] = {c: [] for c in allowed_codes}
    raw_xyz: Dict[str, List[Tuple[float, float, float]]] = {
        c: [] for c in allowed_codes
    }

    stats = {
        "total_lines": 0,
        "kept_rows": 0,
        "malformed_rows": 0,
        "out_of_order_events": 0,
        "duplicate_timestamps": 0,
    }

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stats["total_lines"] += 1
            cleaned = line.strip().rstrip(";").strip()
            if not cleaned:
                continue
            parts = cleaned.split(",")
            if len(parts) != 6:
                stats["malformed_rows"] += 1
                continue
            code = parts[1].strip()
            if code not in allowed_codes:
                continue
            try:
                ts = int(parts[2].strip())
                x = float(parts[3].strip())
                y = float(parts[4].strip())
                z = float(parts[5].strip())
            except ValueError:
                stats["malformed_rows"] += 1
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)) or ts <= 0:
                stats["malformed_rows"] += 1
                continue

            raw_ts[code].append(ts)
            raw_xyz[code].append((x, y, z))
            stats["kept_rows"] += 1

    organized: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for code in allowed_codes:
        if not raw_ts[code]:
            continue
        ts_arr = np.asarray(raw_ts[code], dtype=np.int64)
        xyz_arr = np.asarray(raw_xyz[code], dtype=np.float64)

        # Count out-of-order steps before sorting
        if len(ts_arr) > 1:
            stats["out_of_order_events"] += int(np.sum(np.diff(ts_arr) < 0))

        # Sort chronologically by timestamp
        order = np.argsort(ts_arr, kind="mergesort")
        ts_sorted = ts_arr[order]
        xyz_sorted = xyz_arr[order]

        # Deduplicate identical timestamps by averaging their sensor readings
        unique_ts, inverse_idx, counts = np.unique(
            ts_sorted, return_inverse=True, return_counts=True
        )
        n_dups = int(len(ts_sorted) - len(unique_ts))
        stats["duplicate_timestamps"] += n_dups

        if n_dups > 0:
            xyz_unique = np.zeros((len(unique_ts), 3), dtype=np.float64)
            for axis in range(3):
                np.add.at(xyz_unique[:, axis], inverse_idx, xyz_sorted[:, axis])
            xyz_unique /= counts[:, None]
        else:
            xyz_unique = xyz_sorted

        organized[code] = (unique_ts, xyz_unique)

    return organized, stats


def split_into_contiguous_bouts(
    ts: np.ndarray, xyz: np.ndarray, max_gap_ns: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a sorted time series whenever consecutive timestamps jump by > max_gap_ns."""
    if len(ts) < 2:
        return []
    gaps = np.where(np.diff(ts) > max_gap_ns)[0] + 1
    indices = np.split(np.arange(len(ts)), gaps)
    return [(ts[idx], xyz[idx]) for idx in indices if len(idx) >= 2]


def align_accel_and_gyro_20hz(
    accel_ts: np.ndarray,
    accel_xyz: np.ndarray,
    gyro_ts: np.ndarray,
    gyro_xyz: np.ndarray,
    max_gap_sec: float = 0.5,
) -> List[np.ndarray]:
    """Align sorted accel and gyro recordings onto shared 20 Hz (50 ms) contiguous grids.

    Splits each sensor stream at timestamp gaps > ``max_gap_sec``, finds every
    overlapping bout pair between accelerometer and gyroscope, and linearly
    interpolates ``(ax, ay, az, gx, gy, gz)`` onto a uniform 50 ms grid.
    """
    max_gap_ns = int(max_gap_sec * 1_000_000_000)
    a_bouts = split_into_contiguous_bouts(accel_ts, accel_xyz, max_gap_ns)
    g_bouts = split_into_contiguous_bouts(gyro_ts, gyro_xyz, max_gap_ns)

    aligned_segments: List[np.ndarray] = []
    for a_t, a_v in a_bouts:
        for g_t, g_v in g_bouts:
            t_start = max(int(a_t[0]), int(g_t[0]))
            t_end = min(int(a_t[-1]), int(g_t[-1]))
            if t_end - t_start < STEP_NS * 20:  # Require at least 1 second of overlap
                continue

            n_steps = int((t_end - t_start) // STEP_NS) + 1
            grid = t_start + np.arange(n_steps, dtype=np.int64) * STEP_NS

            # Relative float64 time in seconds for numerical precision during interpolation
            t0 = t_start
            grid_sec = (grid - t0) * 1e-9
            a_sec = (a_t - t0) * 1e-9
            g_sec = (g_t - t0) * 1e-9

            seg = np.empty((n_steps, 6), dtype=np.float32)
            for axis in range(3):
                seg[:, axis] = np.interp(grid_sec, a_sec, a_v[:, axis]).astype(
                    np.float32
                )
                seg[:, axis + 3] = np.interp(grid_sec, g_sec, g_v[:, axis]).astype(
                    np.float32
                )
            aligned_segments.append(seg)

    return aligned_segments


def create_participant_split(
    participants: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    splits_path: Path = SPLITS_PATH,
) -> Dict:
    """Create and save a reproducible participant-disjoint train/val/test split."""
    sorted_subjs = sorted(participants)
    n_total = len(sorted_subjs)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sorted_subjs))

    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    train_subjs = sorted(int(s) for s in shuffled[:n_train])
    val_subjs = sorted(int(s) for s in shuffled[n_train : n_train + n_val])
    test_subjs = sorted(int(s) for s in shuffled[n_train + n_val :])

    split_info = {
        "seed": seed,
        "split_strategy": "participant_disjoint",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "total_participants": n_total,
        "counts": {
            "train": len(train_subjs),
            "val": len(val_subjs),
            "test": len(test_subjs),
        },
        "train_participants": train_subjs,
        "val_participants": val_subjs,
        "test_participants": test_subjs,
    }
    splits_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    return split_info


def segment_into_windows(
    aligned_by_subj_act: Dict[int, Dict[str, List[np.ndarray]]],
    window_sec: int,
    code_to_label: Dict[str, int],
    split_info: Dict,
    overlap: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Slice aligned 20 Hz segments into fixed-duration windows per participant split."""
    win_samples = int(window_sec * SAMPLING_RATE_HZ)
    step_samples = max(1, int(round(win_samples * (1.0 - overlap))))

    subj_to_split: Dict[int, str] = {}
    for s in split_info["train_participants"]:
        subj_to_split[int(s)] = "train"
    for s in split_info["val_participants"]:
        subj_to_split[int(s)] = "val"
    for s in split_info["test_participants"]:
        subj_to_split[int(s)] = "test"

    buckets: Dict[str, Dict[str, List]] = {
        sp: {"X": [], "y": [], "subj": []} for sp in ("train", "val", "test")
    }

    for subj in sorted(aligned_by_subj_act.keys()):
        sp = subj_to_split[subj]
        for code in sorted(aligned_by_subj_act[subj].keys()):
            label = code_to_label[code]
            for seg in aligned_by_subj_act[subj][code]:
                n = seg.shape[0]
                if n < win_samples:
                    continue
                for start in range(0, n - win_samples + 1, step_samples):
                    buckets[sp]["X"].append(seg[start : start + win_samples])
                    buckets[sp]["y"].append(label)
                    buckets[sp]["subj"].append(subj)

    out: Dict[str, np.ndarray] = {}
    for sp in ("train", "val", "test"):
        if buckets[sp]["X"]:
            out[f"X_{sp}"] = np.stack(buckets[sp]["X"], axis=0).astype(np.float32)
            out[f"y_{sp}"] = np.asarray(buckets[sp]["y"], dtype=np.int64)
            out[f"subj_{sp}"] = np.asarray(buckets[sp]["subj"], dtype=np.int64)
        else:
            out[f"X_{sp}"] = np.empty((0, win_samples, 6), dtype=np.float32)
            out[f"y_{sp}"] = np.empty((0,), dtype=np.int64)
            out[f"subj_{sp}"] = np.empty((0,), dtype=np.int64)

    return out


def run_preprocessing(
    windows: List[int],
    overlap: float = 0.0,
    max_gap_sec: float = 0.5,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict:
    """Execute the full cleaning, alignment, participant split, windowing, and feature pipeline."""
    # 1. Load or generate top-5 selected activities
    if not SELECTED_ACTIVITIES_PATH.exists():
        print("selected_activities.json not found; running activity_selection.py (top-5)...")
        selection = select_top_activities(top_k=5)
    else:
        selection = json.loads(SELECTED_ACTIVITIES_PATH.read_text(encoding="utf-8"))

    selected_codes: List[str] = selection["selected_codes"]
    selected_activities: Dict[str, str] = selection["selected_activities"]
    allowed_set = set(selected_codes)

    # Consistent integer class encoding (0..K-1) in rank order of selected_codes
    code_to_label = {code: idx for idx, code in enumerate(selected_codes)}
    label_to_code = {idx: code for idx, code in enumerate(selected_codes)}
    label_to_activity = {
        idx: selected_activities[code] for idx, code in enumerate(selected_codes)
    }

    print(
        "Selected activities: "
        + ", ".join(
            f"{idx}:{code} ({selected_activities[code]})"
            for idx, code in enumerate(selected_codes)
        )
    )

    # 2. Discover participants present in both accel and gyro directories
    accel_files = {
        int(p.stem.split("_")[1]): p
        for p in sorted((RAW_DIR / "accel").glob("data_*_accel_phone.txt"))
    }
    gyro_files = {
        int(p.stem.split("_")[1]): p
        for p in sorted((RAW_DIR / "gyro").glob("data_*_gyro_phone.txt"))
    }
    participants = sorted(set(accel_files.keys()) & set(gyro_files.keys()))
    print(f"Found {len(participants)} participants with both accel and gyro data.")

    # 3. Clean, sort by timestamp, deduplicate, and align onto 20 Hz grid
    aligned_by_subj_act: Dict[int, Dict[str, List[np.ndarray]]] = {}
    agg_stats = {
        "accel_kept_rows": 0,
        "gyro_kept_rows": 0,
        "accel_malformed_rows": 0,
        "gyro_malformed_rows": 0,
        "accel_out_of_order": 0,
        "gyro_out_of_order": 0,
        "accel_duplicate_timestamps": 0,
        "gyro_duplicate_timestamps": 0,
        "aligned_20hz_samples": 0,
        "aligned_contiguous_bouts": 0,
    }

    for subj in participants:
        a_org, a_st = parse_raw_sensor_file(accel_files[subj], allowed_set)
        g_org, g_st = parse_raw_sensor_file(gyro_files[subj], allowed_set)

        agg_stats["accel_kept_rows"] += a_st["kept_rows"]
        agg_stats["gyro_kept_rows"] += g_st["kept_rows"]
        agg_stats["accel_malformed_rows"] += a_st["malformed_rows"]
        agg_stats["gyro_malformed_rows"] += g_st["malformed_rows"]
        agg_stats["accel_out_of_order"] += a_st["out_of_order_events"]
        agg_stats["gyro_out_of_order"] += g_st["out_of_order_events"]
        agg_stats["accel_duplicate_timestamps"] += a_st["duplicate_timestamps"]
        agg_stats["gyro_duplicate_timestamps"] += g_st["duplicate_timestamps"]

        aligned_by_subj_act[subj] = {}
        for code in selected_codes:
            if code not in a_org or code not in g_org:
                continue
            a_ts, a_xyz = a_org[code]
            g_ts, g_xyz = g_org[code]
            segments = align_accel_and_gyro_20hz(
                a_ts, a_xyz, g_ts, g_xyz, max_gap_sec=max_gap_sec
            )
            if segments:
                aligned_by_subj_act[subj][code] = segments
                agg_stats["aligned_contiguous_bouts"] += len(segments)
                agg_stats["aligned_20hz_samples"] += sum(s.shape[0] for s in segments)

    print(
        f"Cleaning & 20 Hz alignment complete: "
        f"{agg_stats['aligned_contiguous_bouts']} contiguous bouts, "
        f"{agg_stats['aligned_20hz_samples']:,} aligned 6-channel time steps."
    )

    # 4. Participant-disjoint train / val / test split
    split_info = create_participant_split(
        participants=participants,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        splits_path=SPLITS_PATH,
    )
    print(
        f"Participant-disjoint split saved to splits.json: "
        f"train={split_info['counts']['train']} users, "
        f"val={split_info['counts']['val']} users, "
        f"test={split_info['counts']['test']} users."
    )

    # 5. Segment into 1s, 2s, 5s, 10s windows and build ML + DL representations
    raw_win_dir = PROCESSED_DIR / "raw_windows"
    raw_win_dir.mkdir(parents=True, exist_ok=True)

    window_summaries: Dict[str, Dict] = {}
    print("\n" + "=" * 88)
    print(
        f"{'Win (s)':<8}{'Samples':<9}{'Train Win':>11}{'Val Win':>10}{'Test Win':>10}"
        f"{'Total Win':>11}{'DL Shape (Train)':>20}{'ML Shape (Train)':>18}"
    )
    print("-" * 88)

    for w in windows:
        splits_data = segment_into_windows(
            aligned_by_subj_act=aligned_by_subj_act,
            window_sec=w,
            code_to_label=code_to_label,
            split_info=split_info,
            overlap=overlap,
        )

        # Save unnormalized raw windows for reference / reproducibility
        raw_npz_path = raw_win_dir / f"windows_{w}s.npz"
        np.savez_compressed(
            raw_npz_path,
            **splits_data,
            channel_names=np.array(CHANNEL_NAMES),
        )

        rep_info = build_representations_for_window(
            window_sec=w,
            splits_data=splits_data,
            idx_to_code=label_to_code,
            idx_to_activity=label_to_activity,
            processed_dir=PROCESSED_DIR,
        )

        n_tr = int(splits_data["y_train"].shape[0])
        n_va = int(splits_data["y_val"].shape[0])
        n_te = int(splits_data["y_test"].shape[0])
        n_tot = n_tr + n_va + n_te

        # Per-class counts per split
        class_counts = {}
        for idx, code in label_to_code.items():
            class_counts[f"{code}_{label_to_activity[idx]}"] = {
                "train": int(np.sum(splits_data["y_train"] == idx)),
                "val": int(np.sum(splits_data["y_val"] == idx)),
                "test": int(np.sum(splits_data["y_test"] == idx)),
                "total": int(
                    np.sum(splits_data["y_train"] == idx)
                    + np.sum(splits_data["y_val"] == idx)
                    + np.sum(splits_data["y_test"] == idx)
                ),
            }

        rep_info["raw_windows_path"] = str(raw_npz_path.relative_to(PROJECT_ROOT))
        rep_info["window_samples"] = int(w * SAMPLING_RATE_HZ)
        rep_info["counts"] = {
            "train": n_tr,
            "val": n_va,
            "test": n_te,
            "total": n_tot,
        }
        rep_info["class_distribution"] = class_counts
        window_summaries[f"{w}s"] = rep_info

        print(
            f"{f'{w}s':<8}{w * SAMPLING_RATE_HZ:<9}{n_tr:>11,}{n_va:>10,}{n_te:>10,}"
            f"{n_tot:>11,}{str(tuple(rep_info['sequence_shape_train'])):>20}"
            f"{str(tuple(rep_info['feature_shape_train'])):>18}"
        )

    print("=" * 88)

    summary = {
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "channels": list(CHANNEL_NAMES),
        "window_durations_sec": windows,
        "window_overlap": overlap,
        "max_gap_sec": max_gap_sec,
        "code_to_label": code_to_label,
        "label_to_code": {str(k): v for k, v in label_to_code.items()},
        "label_to_activity": {str(k): v for k, v in label_to_activity.items()},
        "num_statistical_features": len(get_feature_names()),
        "statistical_feature_names": get_feature_names(),
        "cleaning_stats": agg_stats,
        "participant_split": split_info,
        "windows": window_summaries,
    }

    summary_path = PROCESSED_DIR / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved full preprocessing summary to: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, align, split by participant, window (1s, 2s, 5s, 10s), and extract ML/DL representations."
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Window durations in seconds (default: 1 2 5 10)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Fractional overlap between consecutive windows in [0, 0.9] (default: 0.0 for non-overlapping)",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=0.5,
        help="Maximum timestamp gap (seconds) before splitting into separate contiguous bouts (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for participant-disjoint train/val/test split (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of participants assigned to training set (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of participants assigned to validation set (default: 0.15)",
    )
    args = parser.parse_args()

    run_preprocessing(
        windows=args.windows,
        overlap=args.overlap,
        max_gap_sec=args.max_gap_sec,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
