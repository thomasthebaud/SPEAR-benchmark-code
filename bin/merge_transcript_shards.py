import argparse
from pathlib import Path

import pandas as pd

from transcript_csv_utils import asr_output_name, strip_unnamed_columns


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded ASR transcript CSVs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asr", required=True)
    parser.add_argument("--num-shards", required=True, type=int)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")

    asr_name = asr_output_name(args.asr)
    output_path = args.output_dir / f"{asr_name}_transcripts.csv"
    frames = []

    if output_path.exists():
        frames.append(strip_unnamed_columns(pd.read_csv(output_path)))

    for shard_index in range(args.num_shards):
        shard_path = args.output_dir / f"{asr_name}_transcripts_shard_{shard_index}_of_{args.num_shards}.csv"
        if not shard_path.exists():
            print(f"[WARN] Missing shard transcript CSV: {shard_path}", flush=True)
            continue
        frames.append(strip_unnamed_columns(pd.read_csv(shard_path)))

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        dedupe_column = "answer_audio_path" if "answer_audio_path" in merged.columns else None
        if dedupe_column is None and "audio_path" in merged.columns:
            dedupe_column = "audio_path"
        if dedupe_column is not None:
            merged = merged.drop_duplicates(subset=[dedupe_column], keep="last")
            merged = merged.sort_values(dedupe_column).reset_index(drop=True)
    else:
        merged = pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    merged.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(f"Merged {len(merged)} transcript rows into {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
