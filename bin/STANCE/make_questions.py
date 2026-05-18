#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FILE_ID_RE = re.compile(r"(V\d+_S\d+_I\d+_P[^_./\\]+)", re.I)
TURN_RE = re.compile(r"(?P<speaker>[A-Za-z0-9]+):\s*(?P<text>.*?)(?=\s+[A-Za-z0-9]+:\s*|$)")
SPEARBENCH_DIR = Path(__file__).resolve().parents[3]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text.strip()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_question(path: Path, index: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("outside_judge", [])
    for question in questions:
        if int(question.get("index", -1)) == int(index):
            return question
    raise SystemExit(f"Question index {index} not found in {path}")


def load_role_categories(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    role_to_category: dict[str, str] = {}
    for row in read_csv_rows(path):
        category = safe_str(row.get("category_name"))
        adjectives = safe_str(row.get("Adjectives assigned"))
        for adjective in adjectives.split(","):
            role = adjective.strip()
            if role:
                role_to_category[role.lower()] = category
    return role_to_category


def load_abmapped_assets(assets_dir: Path) -> dict[str, dict[str, str]]:
    path = assets_dir / "interactions_role_ABmapped.csv"
    if not path.exists():
        raise SystemExit(f"Missing asset file: {path}")

    by_file_id: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        for side in ("a", "b"):
            file_id = safe_str(row.get(f"{side}_id"))
            if file_id:
                by_file_id[file_id] = row
    return by_file_id


def load_relationships(assets_dir: Path) -> dict[tuple[str, str], str]:
    path = assets_dir / "relationships.csv"
    if not path.exists():
        return {}

    relationships: dict[tuple[str, str], str] = {}
    for row in read_csv_rows(path):
        vendor = safe_str(row.get("vendor_id"))
        session = safe_str(row.get("session_id"))
        detail = safe_str(row.get("relationship_detail")) or safe_str(row.get("relationship"))
        if vendor and session:
            relationships[(vendor, session)] = detail
    return relationships


def extract_file_ids(row: dict[str, Any]) -> list[str]:
    candidates = [
        safe_str(row.get("audio_path")),
        safe_str(row.get("initial_conversation")),
        safe_str(row.get("utterance_id")),
    ]
    found: list[str] = []
    for text in candidates:
        for match in FILE_ID_RE.findall(text):
            if match not in found:
                found.append(match)
    return found


def get_asset_row(row: dict[str, Any], by_file_id: dict[str, dict[str, str]]) -> dict[str, str]:
    for file_id in extract_file_ids(row):
        if file_id in by_file_id:
            return by_file_id[file_id]
    return {}


def relationship_detail(row: dict[str, Any], asset_row: dict[str, str], relationships: dict[tuple[str, str], str]) -> str:
    conversation = safe_str(row.get("initial_conversation"))
    vendor = conversation.split("_", maxsplit=1)[0]
    session = ""
    if "_S" in conversation:
        session = conversation.split("_S", maxsplit=1)[1].split("_", maxsplit=1)[0]
    return relationships.get((vendor, session), "") or safe_str(asset_row.get("interaction_type"))


def parse_turns(transcript: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for match in TURN_RE.finditer(transcript):
        speaker = safe_str(match.group("speaker"))
        text = safe_str(match.group("text"))
        if speaker and text:
            turns.append((speaker, text))
    return turns


def infer_target_speaker(row: dict[str, Any]) -> str:
    transcript = safe_str(row.get("transcript"))
    before_model_answer = transcript.split("<...>", maxsplit=1)[0]
    turns = parse_turns(before_model_answer or transcript)
    speakers = [s.strip() for s in safe_str(row.get("speakers")).split("|") if s.strip()]

    question_speaker = ""
    for speaker, text in turns:
        if text.rstrip().endswith("?"):
            question_speaker = speaker

    if question_speaker and len(speakers) >= 2:
        for speaker in speakers:
            if speaker != question_speaker:
                return speaker

    if turns:
        return turns[-1][0]
    return speakers[-1] if speakers else ""


def side_for_speaker(speaker: str, asset_row: dict[str, str], row: dict[str, Any]) -> str:
    speaker = safe_str(speaker)
    if speaker and speaker in safe_str(asset_row.get("a_id")):
        return "a"
    if speaker and speaker in safe_str(asset_row.get("b_id")):
        return "b"

    speakers = [s.strip() for s in safe_str(row.get("speakers")).split("|") if s.strip()]
    if speaker and speakers:
        if speaker == speakers[0]:
            return "a"
        if len(speakers) > 1 and speaker == speakers[1]:
            return "b"
    return ""


def category_for_role(role: str, side: str, asset_row: dict[str, str], role_categories: dict[str, str]) -> str:
    explicit = safe_str(asset_row.get(f"category_{side}"))
    if explicit:
        return explicit
    return role_categories.get(safe_str(role).lower(), "")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--questions-csv", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument("--input-mode", choices=("single", "interaction"), required=True)
    parser.add_argument("--assets_dir", "--assets-dir", dest="assets_dir", type=Path, required=True)
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=SPEARBENCH_DIR / "STANCE_outputs/followupGPT/questions_main.json",
        help="STANCE question JSON. Defaults to the checked-in GPT STANCE config.",
    )
    parser.add_argument("--question-index", type=int, required=True)
    parser.add_argument(
        "--category-roles-csv",
        type=Path,
        default=SPEARBENCH_DIR / "STANCE_outputs/followupGPT/category_roles.csv",
    )
    args = parser.parse_args()

    metadata_rows = read_csv_rows(args.metadata)
    question = load_question(args.questions_json, args.question_index)
    by_file_id = load_abmapped_assets(args.assets_dir)
    relationships = load_relationships(args.assets_dir)
    role_categories = load_role_categories(args.category_roles_csv)
    requested_roles = {role.lower() for role in args.roles}
    related_categories = {safe_str(cat).lower() for cat in question.get("related_categories", [])}

    output_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(metadata_rows):
        asset_row = get_asset_row(row, by_file_id)
        target_speaker = infer_target_speaker(row)
        target_side = side_for_speaker(target_speaker, asset_row, row)
        target_role = safe_str(asset_row.get(f"role_{target_side}")) if target_side else ""
        target_category = category_for_role(target_role, target_side, asset_row, role_categories) if target_side else ""
        can_use = (
            safe_str(target_role).lower() in requested_roles
            or safe_str(target_category).lower() in related_categories
        )

        merged = dict(row)
        merged.update(
            {
                "row_idx": row_idx,
                "question_index": int(args.question_index),
                "input_mode": args.input_mode,
                "stance_roles": "|".join(args.roles),
                "stance_related_categories": "|".join(safe_str(c) for c in question.get("related_categories", [])),
                "stance_question": safe_str(question.get("question")),
                "positive_definition": safe_str(question.get("positive_defination")),
                "negative_definition": safe_str(question.get("negative_defination")),
                "positive_followups": json.dumps(question.get("positive_followups", []), ensure_ascii=False),
                "negative_followups": json.dumps(question.get("negative_followups", []), ensure_ascii=False),
                "target_speaker": target_speaker,
                "target_side": target_side.upper(),
                "target_role": target_role,
                "target_category": target_category,
                "can_use_for_stance": bool_text(can_use),
                "participant_a_prompt_text": safe_str(asset_row.get("participant_a_prompt_text")),
                "participant_b_prompt_text": safe_str(asset_row.get("participant_b_prompt_text")),
                "role_a": safe_str(asset_row.get("role_a")),
                "role_b": safe_str(asset_row.get("role_b")),
                "category_a": category_for_role(safe_str(asset_row.get("role_a")), "a", asset_row, role_categories),
                "category_b": category_for_role(safe_str(asset_row.get("role_b")), "b", asset_row, role_categories),
                "relationship_detail": relationship_detail(row, asset_row, relationships),
                "asset_prompt_id_unique": safe_str(asset_row.get("prompt_id_unique")),
                "asset_a_id": safe_str(asset_row.get("a_id")),
                "asset_b_id": safe_str(asset_row.get("b_id")),
            }
        )
        output_rows.append(merged)

    original_fields = list(metadata_rows[0].keys()) if metadata_rows else []
    added_fields = [
        "row_idx",
        "question_index",
        "input_mode",
        "stance_roles",
        "stance_related_categories",
        "stance_question",
        "positive_definition",
        "negative_definition",
        "positive_followups",
        "negative_followups",
        "target_speaker",
        "target_side",
        "target_role",
        "target_category",
        "can_use_for_stance",
        "participant_a_prompt_text",
        "participant_b_prompt_text",
        "role_a",
        "role_b",
        "category_a",
        "category_b",
        "relationship_detail",
        "asset_prompt_id_unique",
        "asset_a_id",
        "asset_b_id",
    ]
    fieldnames = original_fields + [field for field in added_fields if field not in original_fields]
    write_csv_rows(args.questions_csv, output_rows, fieldnames)

    usable = sum(1 for row in output_rows if row["can_use_for_stance"] == "true")
    print(f"Wrote {len(output_rows)} STANCE question rows to {args.questions_csv} ({usable} usable).")


if __name__ == "__main__":
    main()
