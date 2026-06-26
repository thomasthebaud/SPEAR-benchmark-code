import argparse
from pathlib import Path

import pandas as pd

from transcript_csv_utils import COMPLETE_EXIT_CODE, count_missing_transcripts, strip_unnamed_columns


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether ASR transcription shards need to run.")
    parser.add_argument("--input-metadata", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    if not args.input_metadata.exists():
        raise FileNotFoundError(f"Missing input metadata: {args.input_metadata}")

    metadata = pd.read_csv(args.input_metadata)
    if metadata.empty:
        print(f"No input rows in {args.input_metadata}; skipping", flush=True)
        return COMPLETE_EXIT_CODE

    if not args.output_path.exists():
        print(f"No transcript CSV yet: {args.output_path}; launching shards", flush=True)
        return 0

    try:
        transcripts = strip_unnamed_columns(pd.read_csv(args.output_path))
    except Exception as exc:
        print(f"Could not read {args.output_path}: {exc}; launching shards", flush=True)
        return 0

    missing = count_missing_transcripts(metadata, transcripts)
    if missing:
        print(f"{missing}/{len(metadata)} transcript row(s) missing in {args.output_path}; launching shards", flush=True)
        return 0

    print(f"All {len(metadata)} transcript row(s) already present in {args.output_path}; skipping", flush=True)
    return COMPLETE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
