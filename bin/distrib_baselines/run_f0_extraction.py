import csv
import sys
import traceback
from pathlib import Path

import numpy as np
import parselmouth
from tqdm import tqdm

# ======================
# CONFIG
# ======================
ALL_WAVS_TXT = Path("/home/ahallur1/spear/Vox_Profile/vox-profile-release/all_wavs.txt")
RELATIONSHIP_CSV = Path(
    "/export/fs06/corpora8/seamless_interaction/datasets/assets/relationships.csv"
)

OUT_DIR = Path("/home/ahallur1/spear/Seamless_Experiments/F0/shard_csvs")

F0_FLOOR = 75
F0_CEILING = 500
MIN_VOICED_RATIO = 0.05

# ======================
# CSV HEADER
# ======================
CSV_HEADER = [
    # identifiers
    "wav_path",
    "orig_id",
    "vendor_id",
    "session_id",
    "subset",
    "split",
    # relationship
    "relationship",
    "relationship_detail",
    # durations
    "total_duration_s",
    "voiced_duration_s",
    "voiced_ratio",
    "n_voiced_frames",
    # raw F0
    "f0_mean_raw",
    "f0_median_raw",
    "f0_std_raw",
    "f0_min_raw",
    "f0_max_raw",
    "f0_range_raw",
    # 10-90 trimmed
    "f0_p10",
    "f0_p90",
    "f0_range_p10_p90",
    "f0_mean_p10_p90",
    "f0_std_p10_p90",
    # 25-75 trimmed
    "f0_p25",
    "f0_p75",
    "f0_range_p25_p75",
    "f0_mean_p25_p75",
    "f0_std_p25_p75",
    # status
    "status",
]

# ======================
# HELPERS
# ======================
def robust_stats(f0, low_p, high_p):
    lo = np.percentile(f0, low_p)
    hi = np.percentile(f0, high_p)
    trimmed = f0[(f0 >= lo) & (f0 <= hi)]
    if len(trimmed) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    return (
        lo,
        hi,
        hi - lo,
        float(np.mean(trimmed)),
        float(np.std(trimmed)),
    )


def load_relationship_map(relationship_csv: Path):
    relationship_map = {}
    if relationship_csv.exists():
        with open(relationship_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["vendor_id"], row["session_id"])
                relationship_map[key] = (
                    row["relationship"],
                    row["relationship_detail"],
                )
    return relationship_map


def parse_metadata(wav_path: Path, relationship_map):
    parts = wav_path.parts
    subset = parts[-7]
    split = parts[-6]

    fname = wav_path.stem
    # example: V00_S2051_I00001000_P1309A
    vendor_id = fname.split("_")[0]
    session_id = fname.split("_")[1][1:]

    orig_id = fname

    rel, rel_detail = relationship_map.get(
        (vendor_id, session_id), ("UNKNOWN", "UNKNOWN")
    )

    return subset, split, vendor_id, session_id, orig_id, rel, rel_detail


# ======================
# MAIN EXTRACTION
# ======================
def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python run_f0_extraction.py <shard_idx> <num_shards>")
        return 1

    shard_idx = int(sys.argv[1])
    num_shards = int(sys.argv[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"f0_shard_{shard_idx:04d}.csv"

    relationship_map = load_relationship_map(RELATIONSHIP_CSV)

    with open(ALL_WAVS_TXT) as f:
        all_wavs = [line.strip() for line in f if line.strip()]

    shard_wavs = all_wavs[shard_idx :: num_shards]
    print(f"Shard {shard_idx}: processing {len(shard_wavs)} wav files")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for wav_str in tqdm(shard_wavs):
            wav_path = Path(wav_str)
            try:
                subset, split, vendor_id, session_id, orig_id, rel, rel_detail = parse_metadata(
                    wav_path, relationship_map
                )

                snd = parselmouth.Sound(str(wav_path))
                total_dur = snd.get_total_duration()

                pitch = snd.to_pitch_ac(
                    pitch_floor=F0_FLOOR,
                    pitch_ceiling=F0_CEILING,
                )

                f0 = pitch.selected_array["frequency"]
                voiced = f0[f0 > 0]

                total_frames = len(f0)
                voiced_frames = len(voiced)

                voiced_ratio = voiced_frames / total_frames if total_frames else 0.0
                voiced_duration = voiced_ratio * total_dur

                row = {
                    "wav_path": str(wav_path),
                    "orig_id": orig_id,
                    "vendor_id": vendor_id,
                    "session_id": session_id,
                    "subset": subset,
                    "split": split,
                    "relationship": rel,
                    "relationship_detail": rel_detail,
                    "total_duration_s": total_dur,
                    "voiced_duration_s": voiced_duration,
                    "voiced_ratio": voiced_ratio,
                    "n_voiced_frames": voiced_frames,
                    "status": "OK",
                }

                if voiced_frames == 0:
                    row["status"] = "NO_VOICED_FRAMES"
                    writer.writerow(row)
                    continue

                # RAW
                row.update(
                    {
                        "f0_mean_raw": float(np.mean(voiced)),
                        "f0_median_raw": float(np.median(voiced)),
                        "f0_std_raw": float(np.std(voiced)),
                        "f0_min_raw": float(np.min(voiced)),
                        "f0_max_raw": float(np.max(voiced)),
                        "f0_range_raw": float(np.max(voiced) - np.min(voiced)),
                    }
                )

                # 10-90
                p10, p90, r1090, m1090, s1090 = robust_stats(voiced, 10, 90)
                row.update(
                    {
                        "f0_p10": p10,
                        "f0_p90": p90,
                        "f0_range_p10_p90": r1090,
                        "f0_mean_p10_p90": m1090,
                        "f0_std_p10_p90": s1090,
                    }
                )

                # 25-75
                p25, p75, r2575, m2575, s2575 = robust_stats(voiced, 25, 75)
                row.update(
                    {
                        "f0_p25": p25,
                        "f0_p75": p75,
                        "f0_range_p25_p75": r2575,
                        "f0_mean_p25_p75": m2575,
                        "f0_std_p25_p75": s2575,
                    }
                )

                if voiced_ratio < MIN_VOICED_RATIO:
                    row["status"] = "LOW_VOICED_RATIO"

                writer.writerow(row)

            except Exception as e:
                writer.writerow(
                    {
                        "wav_path": str(wav_path),
                        "orig_id": wav_path.stem,
                        "status": f"ERROR: {e}",
                    }
                )
                traceback.print_exc()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
