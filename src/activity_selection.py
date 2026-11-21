"""Select the top-k activities with the most raw samples across participants.

Counts valid sensor rows per activity code across ``data/raw/accel/`` and
``data/raw/gyro/``, ranks activities by total sample count, prints a summary
table, and saves the selected top-k activities (default: k=5) to
``data/processed/selected_activities.json`` for downstream preprocessing.

Usage examples (run from the project root directory):
    # 1. Default: select top 5 activities ranked by total samples (accel + gyro)
    python src/activity_selection.py

    # 2. Select a different number of top activities (e.g., top 6)
    python src/activity_selection.py --top-k 6

    # 3. Rank activities using only accelerometer or only gyroscope sample counts
    python src/activity_selection.py --rank-by accel
    python src/activity_selection.py --rank-by gyro

    # 4. Save the output JSON to a custom path
    python src/activity_selection.py --top-k 5 --rank-by total --output data/processed/my_activities.json

    # 5. View all command-line options
    python src/activity_selection.py --help
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ACTIVITY_KEY_PATH = RAW_DIR / "activity_key.txt"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "selected_activities.json"


def load_activity_key(key_path: Path = ACTIVITY_KEY_PATH) -> Dict[str, str]:
    """Return mapping from activity letter code (e.g. 'A') to name (e.g. 'walking')."""
    mapping: Dict[str, str] = {}
    if not key_path.exists():
        return mapping
    for raw_line in key_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        name, code = [part.strip() for part in line.split("=", 1)]
        mapping[code] = name
    return mapping


def count_sensor_samples(sensor_dir: Path) -> Tuple[Counter, Dict[str, set]]:
    """Count valid rows per activity code and track unique participants per activity."""
    counts: Counter = Counter()
    participants_per_activity: Dict[str, set] = {}

    for txt_path in sorted(sensor_dir.glob("data_*.txt")):
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().rstrip(";").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                subj, code = parts[0].strip(), parts[1].strip()
                if not code:
                    continue
                counts[code] += 1
                if code not in participants_per_activity:
                    participants_per_activity[code] = set()
                participants_per_activity[code].add(subj)

    return counts, participants_per_activity


def select_top_activities(
    raw_dir: Path = RAW_DIR,
    top_k: int = 5,
    rank_by: str = "total",
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Dict:
    """Rank activities by sample count and save top-k selection to JSON.

    Args:
        raw_dir: Path to ``data/raw`` containing ``accel/``, ``gyro/``, and ``activity_key.txt``.
        top_k: Number of top activities to select.
        rank_by: One of ``'total'`` (accel + gyro), ``'accel'``, or ``'gyro'``.
        output_path: Destination JSON path for the selected activities.
    """
    activity_names = load_activity_key(raw_dir / "activity_key.txt")
    accel_counts, accel_subjs = count_sensor_samples(raw_dir / "accel")
    gyro_counts, gyro_subjs = count_sensor_samples(raw_dir / "gyro")

    all_codes = sorted(set(accel_counts.keys()) | set(gyro_counts.keys()))
    rows: List[Dict] = []
    for code in all_codes:
        a_cnt = accel_counts.get(code, 0)
        g_cnt = gyro_counts.get(code, 0)
        total_cnt = a_cnt + g_cnt
        n_accel_subj = len(accel_subjs.get(code, set()))
        n_gyro_subj = len(gyro_subjs.get(code, set()))
        rows.append(
            {
                "code": code,
                "activity": activity_names.get(code, f"unknown_{code}"),
                "accel_samples": a_cnt,
                "gyro_samples": g_cnt,
                "total_samples": total_cnt,
                "accel_participants": n_accel_subj,
                "gyro_participants": n_gyro_subj,
            }
        )

    sort_key = {
        "total": lambda r: (r["total_samples"], r["accel_samples"]),
        "accel": lambda r: (r["accel_samples"], r["total_samples"]),
        "gyro": lambda r: (r["gyro_samples"], r["total_samples"]),
    }[rank_by]

    rows.sort(key=sort_key, reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["selected"] = idx <= top_k

    selected = rows[:top_k]

    result = {
        "top_k": top_k,
        "ranked_by": rank_by,
        "selected_codes": [r["code"] for r in selected],
        "selected_activities": {r["code"]: r["activity"] for r in selected},
        "selected_details": selected,
        "all_activities_ranked": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def print_summary(result: Dict) -> None:
    """Print formatted table of all activities and highlight the top-k selection."""
    header = (
        f"{'Rank':<6}{'Code':<6}{'Activity':<14}"
        f"{'Accel Samples':>15}{'Gyro Samples':>15}{'Total Samples':>15}"
        f"{'Subj (A/G)':>13}{'Selected':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in result["all_activities_ranked"]:
        subj_str = f"{r['accel_participants']}/{r['gyro_participants']}"
        sel_str = "YES *" if r["selected"] else "no"
        print(
            f"{r['rank']:<6}{r['code']:<6}{r['activity']:<14}"
            f"{r['accel_samples']:>15,}{r['gyro_samples']:>15,}{r['total_samples']:>15,}"
            f"{subj_str:>13}{sel_str:>10}"
        )
    print("-" * len(header))
    chosen = ", ".join(
        f"{c} ({name})" for c, name in result["selected_activities"].items()
    )
    print(f"Top {result['top_k']} selected activities (ranked by {result['ranked_by']}): {chosen}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the top-k activities with the most samples in WISDM smartphone raw data."
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top activities to select (default: 5)")
    parser.add_argument(
        "--rank-by",
        choices=["total", "accel", "gyro"],
        default="total",
        help="Metric used to rank activities (default: total = accel + gyro)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file path (default: data/processed/selected_activities.json)",
    )
    args = parser.parse_args()

    result = select_top_activities(top_k=args.top_k, rank_by=args.rank_by, output_path=args.output)
    print_summary(result)
    print(f"\nSaved selection to: {args.output}")


if __name__ == "__main__":
    main()
