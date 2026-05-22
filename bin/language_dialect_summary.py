import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


FAILED_VALUES = {"", "nan", "none", "unknown", "unknown - too short", "unknown - low confidence"}


LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "ar": "arabic",
    "ara": "arabic",
    "zh": "chinese",
    "zho": "chinese",
    "chi": "chinese",
    "haw": "hawaiian",
    "dan": "danish",
    "nld": "dutch",
    "cym": "welsh",
    "mri": "maori",
}


def read_csv(path: Path) -> list:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_language(value: object) -> str:
    normalized = normalize_value(value).replace("_", " ").replace("-", " ")
    return LANGUAGE_ALIASES.get(normalized, normalized)


def is_failed(value: object) -> bool:
    return normalize_value(value) in FAILED_VALUES


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * count / total


def percent(count: int, total: int) -> str:
    return f"{pct(count, total):.2f}%"


def serialize_counts(counts: Counter) -> str:
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def serialize_nested_counts(counts_by_group: dict) -> str:
    output = []
    for group in sorted(counts_by_group):
        output.append(f"{group}={serialize_counts(counts_by_group[group])}")
    return "|".join(output)


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def ranked_language_columns() -> dict:
    columns = {}
    for rank in (2, 3):
        suffix = "nd" if rank == 2 else "rd" if rank == 3 else "th"
        columns[f"{rank}{suffix}_most_predicted"] = ""
        columns[f"{rank}{suffix}_percent"] = "0.00%"
    return columns


def summarize_language(language_csv: Path) -> dict:
    summary = {
        "language_total": 0,
        "language_failed_count": 0,
        "language_failed_percent": "0.00%",
        "english_count": 0,
        "english_percent": "0.00%",
        **ranked_language_columns(),
        "other_count": 0,
        "other_percent": "0.00%",
    }

    if not language_csv.exists():
        return summary

    rows = read_csv(language_csv)
    if rows and "language" not in rows[0]:
        return summary

    total = len(rows)
    failed_count = 0
    english = 0
    non_english_languages = Counter()

    for row in rows:
        language = row.get("language", "")
        normalized = normalize_language(language)
        if is_failed(language):
            failed_count += 1
        elif normalized == "english":
            english += 1
        else:
            non_english_languages[normalized or "unknown"] += 1

    ranked_languages = sorted(
        non_english_languages.items(),
        key=lambda item: (-item[1], item[0]),
    )[:2]
    ranked_columns = ranked_language_columns()
    for rank, (language, count) in zip((2, 3), ranked_languages):
        suffix = "nd" if rank == 2 else "rd" if rank == 3 else "th"
        ranked_columns[f"{rank}{suffix}_most_predicted"] = f"{language}:{count}"
        ranked_columns[f"{rank}{suffix}_percent"] = percent(count, total)

    other = sum(non_english_languages.values())
    summary.update(
        {
            "language_total": total,
            "language_failed_count": failed_count,
            "language_failed_percent": percent(failed_count, total),
            "english_count": english,
            "english_percent": percent(english, total),
            **ranked_columns,
            "other_count": other,
            "other_percent": percent(other, total),
        }
    )
    return summary


def summarize_dialect(dialect_csv: Path) -> dict:
    summary = {
        "dialect_total": 0,
        "dialect_failed_count": 0,
        "dialect_failed_percent": "0.00%",
        "dialect_failed_by_language": "",
    }

    if not dialect_csv.exists():
        return summary

    rows = read_csv(dialect_csv)
    if rows and "dialect" not in rows[0]:
        return summary

    grouped = {}
    for row in rows:
        language = normalize_language(row.get("language", "all")) or "unknown"
        grouped.setdefault(language, []).append(row)

    failed_by_language = Counter()
    dialect_counts_by_language = defaultdict(Counter)

    for language, group in grouped.items():
        for row in group:
            dialect = row.get("dialect", "")
            if is_failed(dialect):
                failed_by_language[language] += 1
            else:
                dialect_counts_by_language[language][str(dialect).strip()] += 1

    failed_count = sum(failed_by_language.values())
    summary.update(
        {
            "dialect_total": len(rows),
            "dialect_failed_count": failed_count,
            "dialect_failed_percent": percent(failed_count, len(rows)),
            "dialect_failed_by_language": serialize_counts(failed_by_language),
        }
    )
    return summary


def summarize(language_csv: Path, dialect_csv: Path) -> dict:
    return {
        **summarize_language(language_csv),
        **summarize_dialect(dialect_csv),
    }


def summarize_combination(results_dir: Path, split: str, subset: str, model: str) -> dict:
    output_dir = results_dir / model / split / subset
    return {
        "split": split,
        "subset": subset,
        "model": model if '-preview' not in model else model.split("-preview")[0],
        **summarize(
            output_dir / "language_id.csv",
            output_dir / "dialect_id.csv",
        ),
    }


def summarize_all(results_dir: Path, splits: list[str], subsets: list[str], models: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        for subset in subsets:
            for model in models:
                rows.append(summarize_combination(results_dir, split, subset, model))
    return rows


def print_table(rows: list[dict], include_header: bool = True) -> None:
    if not rows:
        return

    headers = list(rows[0])
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    separator_after = {
        "model",
        "language_failed_percent",
        "other_percent",
    }

    def format_row(row: dict) -> str:
        parts = []
        for header in headers:
            parts.append(str(row[header]).ljust(widths[header]))
            if header in separator_after:
                parts.append("|")
        return "  ".join(parts)

    if include_header:
        print(format_row({header: header for header in headers}))

    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing model/split/subset result folders.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to summarize, for example: original $llm_model.")
    parser.add_argument("--splits", nargs="+", default=["test", "dev"])
    parser.add_argument("--subsets", nargs="+", default=["improvised", "naturalistic"])
    parser.add_argument("--no-header", action="store_true", help="Do not print the table header.")
    args = parser.parse_args()

    rows = summarize_all(
        Path(args.results_dir),
        args.splits,
        args.subsets,
        unique_values(args.models),
    )
    print_table(rows, include_header=not args.no_header)
