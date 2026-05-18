#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


FILE_ID_RE = re.compile(r"(V\d+_S\d+_I\d+_P\d+A?)")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def atomic_save_pickle(dst: Path, obj: Any) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    with tmp.open("wb") as outfile:
        pickle.dump(obj, outfile)
    os.replace(str(tmp), str(dst))


def hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    text = (text or "").strip().lower()
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    vector = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    vector /= np.linalg.norm(vector) + 1e-6
    return vector


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def unique_preserve_order(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        text = safe_str(text)
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def load_abmapped_assets(assets_dir: Path) -> dict[str, dict]:
    path = assets_dir / "interactions_role_ABmapped.csv"
    rows = read_csv_rows(path)
    by_file_id: dict[str, dict] = {}
    for row in rows:
        for side in ("a", "b"):
            file_id = safe_str(row.get(f"{side}_id"))
            if not file_id:
                continue
            by_file_id[file_id] = row
    return by_file_id


def load_relationship_assets(assets_dir: Path) -> dict[tuple[str, str], str]:
    path = assets_dir / "relationships.csv"
    if not path.exists():
        return {}
    rels: dict[tuple[str, str], str] = {}
    for row in read_csv_rows(path):
        key = (safe_str(row.get("vendor_id")), safe_str(row.get("session_id")))
        detail = safe_str(row.get("relationship_detail")) or safe_str(row.get("relationship"))
        if key[0] and key[1]:
            rels[key] = detail
    return rels


def extract_file_ids(row: dict) -> list[str]:
    candidates = [
        safe_str(row.get("audio_path")),
        safe_str(row.get("initial_conversation")),
    ]
    found: list[str] = []
    for text in candidates:
        for match in FILE_ID_RE.findall(text):
            if match not in found:
                found.append(match)
    return found


def get_asset_row(row: dict, by_file_id: dict[str, dict]) -> dict:
    for file_id in extract_file_ids(row):
        if file_id in by_file_id:
            return by_file_id[file_id]
    return {}


def relationship_detail(row: dict, asset_row: dict, relationship_assets: dict[tuple[str, str], str]) -> str:
    vendor = safe_str(row.get("initial_conversation")).split("_", maxsplit=1)[0]
    session = ""
    if "_S" in safe_str(row.get("initial_conversation")):
        session = safe_str(row.get("initial_conversation")).split("_S", maxsplit=1)[1].split("_", maxsplit=1)[0]

    return (
        relationship_assets.get((vendor, session), "")
        or safe_str(asset_row.get("interaction_type"))
        or safe_str(row.get("initial_conversation"))
    )


def context_text(asset_row: dict, row: dict) -> str:
    prompt_a = safe_str(asset_row.get("participant_a_prompt_text"))
    prompt_b = safe_str(asset_row.get("participant_b_prompt_text"))
    if prompt_a or prompt_b:
        return f"{prompt_a} [SEP] {prompt_b}".strip()
    return safe_str(row.get("transcript"))


def rel_text(row: dict, asset_row: dict, relationship_assets: dict[tuple[str, str], str], template: str) -> str:
    role_a = safe_str(asset_row.get("role_a"))
    role_b = safe_str(asset_row.get("role_b"))

    if (not role_a or not role_b) and row.get("speakers"):
        speakers = safe_str(row.get("speakers")).split("|", maxsplit=1)
        role_a = role_a or (speakers[0].strip() if speakers else "")
        role_b = role_b or (speakers[1].strip() if len(speakers) > 1 else "")

    return template.format(
        speaker_a_role=role_a,
        speaker_b_role=role_b,
        rel_detail=relationship_detail(row, asset_row, relationship_assets),
    ).strip()


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_len: int,
    use_fp16: bool,
    fallback_dim: int,
) -> dict[str, np.ndarray]:
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embed_map: dict[str, np.ndarray] = {}
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {model_name}"):
        batch = texts[start:start + batch_size]
        try:
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", enabled=(use_fp16 and device.type == "cuda")):
                output = model(**encoded)
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=-1)
            for text, embedding in zip(batch, pooled.float().cpu().numpy()):
                embed_map[text] = embedding.astype(np.float32, copy=False)
        except Exception as exc:
            print(f"[TEXT] HF encode failed for one batch ({exc}); using hash fallback.")
            for text in batch:
                embed_map[text] = hash_embedding(text, dim=fallback_dim)
    return embed_map


def build_cache(
    texts: list[str],
    cache_path: Path,
    model_name: str,
    batch_size: int,
    max_len: int,
    use_fp16: bool,
    fallback_dim: int,
) -> dict[str, np.ndarray]:
    unique_texts = unique_preserve_order(texts)
    print(f"[TEXT] unique strings for {cache_path.name}: {len(unique_texts)}")
    try:
        embed_map = encode_texts(unique_texts, model_name, batch_size, max_len, use_fp16, fallback_dim)
    except Exception as exc:
        print(f"[TEXT] HF setup failed ({exc}); using hash fallback.")
        embed_map = {text: hash_embedding(text, dim=fallback_dim) for text in unique_texts}
    atomic_save_pickle(cache_path, embed_map)
    print(f"[TEXT] saved {len(embed_map)} entries to {cache_path}")
    return embed_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV.")
    parser.add_argument("--assets-dir", type=Path, required=True, help="Absolute path to seamless assets directory.")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rel-text-template", type=str, default="{speaker_a_role} [SEP] {speaker_b_role} [SEP] {rel_detail}")
    parser.add_argument("--context-cache-name", type=str, default="context_hf_cache.pkl")
    parser.add_argument("--rel-cache-name", type=str, default="relationship_hf_cache.pkl")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--text-fp16", action="store_true")
    parser.add_argument("--fallback-dim", type=int, default=384)
    args = parser.parse_args()

    metadata_rows = read_csv_rows(args.metadata)
    by_file_id = load_abmapped_assets(args.assets_dir)
    relationship_assets = load_relationship_assets(args.assets_dir)

    context_texts: list[str] = []
    relationship_texts: list[str] = []
    missing_assets = 0
    for row in metadata_rows:
        asset_row = get_asset_row(row, by_file_id)
        if not asset_row:
            missing_assets += 1
        context_texts.append(context_text(asset_row, row))
        relationship_texts.append(rel_text(row, asset_row, relationship_assets, args.rel_text_template))

    if missing_assets:
        print(f"[WARN] No AB-mapped asset row found for {missing_assets}/{len(metadata_rows)} metadata rows; used fallbacks.")

    output_dir = args.metadata.parent
    build_cache(
        context_texts,
        output_dir / args.context_cache_name,
        args.text_model,
        args.batch_size,
        args.max_len,
        args.text_fp16,
        args.fallback_dim,
    )
    build_cache(
        relationship_texts,
        output_dir / args.rel_cache_name,
        args.text_model,
        args.batch_size,
        args.max_len,
        args.text_fp16,
        args.fallback_dim,
    )
