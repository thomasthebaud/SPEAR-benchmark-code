#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from . import infer as infer_module
    from .infer import build_embed_index, load_model_from_checkpoint, predict_csv
    from .extract_pickles import (
        context_text,
        get_asset_row,
        load_abmapped_assets,
        load_relationship_assets,
        rel_text,
        safe_str,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import infer as infer_module
    from infer import build_embed_index, load_model_from_checkpoint, predict_csv
    from extract_pickles import (
        context_text,
        get_asset_row,
        load_abmapped_assets,
        load_relationship_assets,
        rel_text,
        safe_str,
    )


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_pickle_dir(metadata_path: Path) -> Path:
    parts = list(metadata_path.parts)
    if "outputs" in parts:
        idx = parts.index("outputs")
        if len(parts) > idx + 3:
            # .../<protocol>/outputs/<model>/<split>/<subset>/<csv>
            return Path(*parts[:idx]) / "inputs" / parts[idx + 2] / parts[idx + 3]
    return metadata_path.parent


def find_context_metadata(pickle_dir: Path) -> Optional[Path]:
    matches = sorted(pickle_dir.glob("*_metadata.csv"))
    return matches[0] if matches else None


def load_pickle(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        print(f"[WARN] cache not found, using empty map: {path}")
        return {}
    with path.open("rb") as infile:
        obj = pickle.load(infile)
    print(f"Loaded {path.name}: {len(obj)} entries from {path}")
    return obj


def get_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scoring_csv(
    metadata_path: Path,
    context_metadata_path: Optional[Path],
    output_csv: Path,
    info: dict,
    assets_dir: Optional[Path],
    rel_template: str,
) -> Path:
    rows, _ = read_rows(metadata_path)
    context_rows = rows
    if context_metadata_path is not None and context_metadata_path.exists():
        context_rows, _ = read_rows(context_metadata_path)
        print(f"Using text keys from: {context_metadata_path}")
    context_by_stem = {
        Path(safe_str(row.get("audio_path"))).stem: row
        for row in context_rows
        if safe_str(row.get("audio_path"))
    }
    hp = info.get("hyperparams", {}) or {}

    p1_col = hp.get("p1_path_col", "participant1_relpath_abs")
    p2_col = hp.get("p2_path_col", "participant2_relpath_abs")
    label_col = hp.get("label_col", "naturalness")
    context_col = hp.get("context_col", "high_level_context")
    speaker_a_role_col = hp.get("speaker_a_role_col", "speaker_a_role")
    speaker_b_role_col = hp.get("speaker_b_role_col", "speaker_b_role")
    rel_detail_col = hp.get("rel_detail_col", "rel_detail")

    by_file_id = {}
    relationship_assets = {}
    if assets_dir is not None:
        by_file_id = load_abmapped_assets(assets_dir)
        relationship_assets = load_relationship_assets(assets_dir)

    scoring_rows: list[dict] = []
    missing_assets = 0
    for idx, row in enumerate(rows):
        audio_path = Path(safe_str(row.get("audio_path")))
        stem = audio_path.stem
        text_row = context_by_stem.get(stem, context_rows[idx] if idx < len(context_rows) else row)
        asset_row = get_asset_row(text_row, by_file_id) if by_file_id else {}
        if assets_dir is not None and not asset_row:
            missing_assets += 1

        speakers = safe_str(text_row.get("speakers"))
        speaker_a = ""
        speaker_b = ""
        if speakers:
            parts = speakers.split("|", maxsplit=1)
            speaker_a = parts[0].strip()
            speaker_b = parts[1].strip() if len(parts) > 1 else ""

        high_context = context_text(asset_row, text_row)
        relation_text = rel_text(text_row, asset_row, relationship_assets, rel_template)
        rel_detail = relation_text.split("[SEP]")[-1].strip() if "[SEP]" in relation_text else relation_text

        scoring_rows.append(
            {
                "row_idx": idx,
                "audio_path": safe_str(row.get("audio_path")),
                "utterance_id": stem,
                p1_col: stem,
                p2_col: stem,
                label_col: safe_str(row.get(label_col, "")),
                context_col: high_context,
                speaker_a_role_col: safe_str(asset_row.get("role_a")) or speaker_a,
                speaker_b_role_col: safe_str(asset_row.get("role_b")) or speaker_b,
                rel_detail_col: rel_detail,
                "augmentation_type": safe_str(row.get("initial_conversation", "")),
            }
        )

    if missing_assets:
        print(f"[WARN] No asset row found for {missing_assets}/{len(rows)} rows; used metadata fallbacks.")

    fieldnames = [
        "row_idx",
        "audio_path",
        "utterance_id",
        p1_col,
        p2_col,
        label_col,
        context_col,
        speaker_a_role_col,
        speaker_b_role_col,
        rel_detail_col,
        "augmentation_type",
    ]
    write_rows(output_csv, scoring_rows, fieldnames)
    print(f"Prepared naturalness inference CSV: {output_csv}")
    return output_csv


def save_scores(metadata_path: Path, results: list[dict], output_csv: Path) -> None:
    metadata_rows, metadata_fields = read_rows(metadata_path)
    by_idx = {int(result["row_idx"]): result for result in results}

    output_fields = list(metadata_fields)
    for field in ["naturalness_probability", "naturalness_prediction", "naturalness_logit", "naturalness_error"]:
        if field not in output_fields:
            output_fields.append(field)

    output_rows: list[dict] = []
    for idx, row in enumerate(metadata_rows):
        out = dict(row)
        result = by_idx.get(idx, {})
        out["naturalness_probability"] = result.get("probability", "")
        out["naturalness_prediction"] = result.get("prediction", "")
        out["naturalness_logit"] = result.get("logit", "")
        out["naturalness_error"] = result.get("error", "missing_result")
        output_rows.append(out)

    write_rows(output_csv, output_rows, output_fields)
    print(f"Saved utterance naturalness scores: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score benchmark utterances with the naturalness model.")
    parser.add_argument("--metadata", type=Path, required=True, help="Benchmark metadata CSV to score.")
    parser.add_argument("--model-path", type=Path, default=Path("models/naturalness/last_model.pt"))
    parser.add_argument("--outputs", type=Path, required=True, help="Output directory for score CSVs.")
    parser.add_argument("--pickle-dir", type=Path, default=None, help="Directory containing context_hf_cache.pkl and relationship_hf_cache.pkl.")
    parser.add_argument("--assets-dir", type=Path, default=None, help="Seamless assets dir, used only to rebuild inference CSV text keys.")
    parser.add_argument("--embed-root", type=Path, default=None, help="Precomputed naturalness feature root. Defaults to metadata_dir/naturalness/voxprofile_features.")
    parser.add_argument("--context-cache", type=Path, default=None)
    parser.add_argument("--rel-cache", type=Path, default=None)
    parser.add_argument("--context-metadata", type=Path, default=None, help="Metadata CSV used when the text caches were built.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--min-chunks", type=int, default=1, help="Minimum feature chunks required per utterance.")
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    infer_module.MIN_REAL_PAIRS = max(1, int(args.min_chunks))

    print(f"Loading checkpoint: {args.model_path}")
    model, info = load_model_from_checkpoint(str(args.model_path), device)
    model.eval()

    hp = info.get("hyperparams", {}) or {}
    rel_template = hp.get("rel_text_template", "{speaker_a_role} [SEP] {speaker_b_role} [SEP] {rel_detail}")

    embed_root = args.embed_root or (args.metadata.parent / "naturalness" / "voxprofile_features")
    if not embed_root.exists():
        raise FileNotFoundError(f"Precomputed naturalness feature root not found: {embed_root}")
    print(f"Scanning precomputed feature root: {embed_root}")
    embed_index = build_embed_index(embed_root)
    print(f"  base_ids={len(embed_index)} total_chunks={sum(len(v) for v in embed_index.values())}")

    pickle_dir = args.pickle_dir or infer_pickle_dir(args.metadata)
    context_metadata = args.context_metadata or find_context_metadata(pickle_dir)
    context_cache = args.context_cache or (pickle_dir / "context_hf_cache.pkl")
    rel_cache = args.rel_cache or (pickle_dir / "relationship_hf_cache.pkl")

    context_embed_map: Dict[str, np.ndarray] = {}
    rel_embed_map: Dict[str, np.ndarray] = {}
    if info.get("use_aux", False):
        context_embed_map = load_pickle(context_cache)
        rel_embed_map = load_pickle(rel_cache)

    max_pairs = None
    max_seq_len = int(args.max_seq_len or hp.get("max_seq_len", 512) or 0)
    if max_seq_len > 0:
        max_pairs = max(1, max_seq_len // 2)

    scoring_input = args.outputs / "naturalness_inference_input.csv"
    build_scoring_csv(args.metadata, context_metadata, scoring_input, info, args.assets_dir, rel_template)

    results = predict_csv(
        model=model,
        info=info,
        input_csv=str(scoring_input),
        embed_index=embed_index,
        device=device,
        context_embed_map=context_embed_map,
        rel_embed_map=rel_embed_map,
        cache_root=None,
        max_pairs=max_pairs,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    raw_predictions = args.outputs / "naturalness_predictions_raw.csv"
    if results:
        write_rows(raw_predictions, results, list(results[0].keys()))
        print(f"Saved raw infer.py-style predictions: {raw_predictions}")

    score_csv = args.outputs / "naturalness_scores.csv"
    save_scores(args.metadata, results, score_csv)

    n_ok = sum(1 for result in results if result.get("error") in (None, ""))
    print(f"Scored {n_ok}/{len(results)} utterances")


if __name__ == "__main__":
    main()
