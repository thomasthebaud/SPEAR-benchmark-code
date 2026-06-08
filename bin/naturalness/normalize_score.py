#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sigmoid(values: pd.Series) -> pd.Series:
    arr = values.astype(float).to_numpy(copy=True)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return pd.Series(out, index=values.index)


def load_logits(path: Path, column: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing score CSV: {path}")
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise ValueError(f"{path} does not contain required column: {column}")
    logits = numeric(frame[column]).dropna()
    if logits.empty:
        raise ValueError(f"{path} does not contain any valid {column} values")
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize naturalness logits by shifting them so a reference dev set is mostly positive."
    )
    parser.add_argument("--ref-dir", type=Path, required=True, help="Reference directory, usually results/.../original/dev/<subset>.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory to normalize, usually results/.../<model>/test/<subset>.")
    parser.add_argument("--input-name", default="naturalness_scores.csv")
    parser.add_argument("--output-name", default="naturalness_scores_normalized.csv")
    parser.add_argument("--logit-column", default="naturalness_logit")
    parser.add_argument("--positive-fraction", type=float, default=0.99, help="Target fraction of reference logits that should be positive after shifting.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Small offset so the selected reference quantile is strictly positive.")
    args = parser.parse_args()

    if not 0.0 < args.positive_fraction < 1.0:
        parser.error("--positive-fraction must be between 0 and 1")

    ref_path = args.ref_dir / args.input_name
    data_path = args.data_dir / args.input_name
    output_path = args.data_dir / args.output_name

    ref_logits = load_logits(ref_path, args.logit_column)
    reference_quantile = 1.0 - args.positive_fraction
    reference_value = float(np.quantile(ref_logits.to_numpy(dtype=np.float64), reference_quantile))
    shift = -reference_value + float(args.epsilon)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing score CSV: {data_path}")
    frame = pd.read_csv(data_path)
    if args.logit_column not in frame.columns:
        raise ValueError(f"{data_path} does not contain required column: {args.logit_column}")

    raw_logits = numeric(frame[args.logit_column])
    valid = raw_logits.notna()
    shifted_logits = raw_logits + shift

    frame[f"{args.logit_column}_raw"] = raw_logits
    frame["naturalness_logit_shift"] = shift
    frame["naturalness_reference_quantile"] = reference_quantile
    frame["naturalness_reference_logit"] = reference_value
    frame["naturalness_positive_fraction_target"] = args.positive_fraction
    frame.loc[valid, args.logit_column] = shifted_logits.loc[valid]

    if "naturalness_probability" in frame.columns:
        frame.loc[valid, "naturalness_probability"] = sigmoid(shifted_logits.loc[valid])
    if "naturalness_prediction" in frame.columns:
        frame.loc[valid, "naturalness_prediction"] = (shifted_logits.loc[valid] >= 0).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    ref_positive = float(((ref_logits + shift) > 0).mean())
    data_valid = int(valid.sum())
    print(
        f"Wrote {output_path} with shift={shift:.6f}; "
        f"reference_positive_fraction={ref_positive:.4f}; normalized_rows={data_valid}/{len(frame)}"
    )


if __name__ == "__main__":
    main()
