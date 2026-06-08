import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def wav_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def get_lens(relpaths: pd.Series, data_dir: Path) -> np.ndarray:
    lens = []
    for path in relpaths:
        audio_path = (data_dir / Path(path[3:])).resolve()
        lens.append(wav_duration(audio_path))
    return np.array(lens)


def prepared_dialogue_ids(metadata: pd.DataFrame) -> set[str]:
    if "conversation_id" in metadata.columns:
        return set(metadata["conversation_id"].dropna().astype(str))
    return set()


def summarize_split_subset(
    answers_output_dir: Path,
    original_data_csv: Path,
    data_dir: Path,
    split: str,
    subset: str,
) -> dict[str, float]:
    dyads = pd.read_csv(original_data_csv)
    dyads = dyads[dyads["participant1_relpath"].str.contains(f"/{subset}/{split}/", regex=False)]

    lens = np.concatenate(
        [
            get_lens(dyads["participant1_relpath"], data_dir)[np.newaxis, :],
            get_lens(dyads["participant2_relpath"], data_dir)[np.newaxis, :],
        ],
        axis=0,
    )
    lens = np.max(lens, axis=0)

    metadata = pd.read_csv(answers_output_dir / "metadata.csv")
    original_dialogue_ids = [i.split("/")[-1].split("_P")[0] for i in dyads["participant1_relpath"].values]
    prepared_ids = prepared_dialogue_ids(metadata)
    question_hours = metadata["question_end_time"].sum() / 3600.0
    answer_hours = metadata["answer_duration"].sum() / 3600.0

    return {
        "original_sets": len(dyads),
        "original_hours": lens.sum() / 3600.0,
        "used_original_dialogues": len(prepared_ids),
        "unique_original_dialogues": len(set(original_dialogue_ids)),
        "selected_dialogues": len(metadata),
        "question_hours": question_hours,
        "answer_hours": answer_hours,
        "total_hours": question_hours + answer_hours,
        "files_per_dialogue": len(metadata) / len(prepared_ids) if prepared_ids else 0.0,
    }


def print_text_summary(summary: dict[str, float], split: str, subset: str) -> None:
    print(f"For split '{split}' and subset '{subset}':")
    print(f"Found {summary['original_sets']:.0f} dialogues in the original data for the specified split and subset.")
    print(f"Total hours = {summary['original_hours']:.2f}")
    used = summary["used_original_dialogues"]
    total = summary["unique_original_dialogues"]
    percent = 100.0 * used / total if total else 0.0
    print(f"Used {used:.0f}/{total:.0f} dialogues in the prepared data. ({percent:.2f}%)")
    print(f"Produced {summary['selected_dialogues']:.0f} audio files {summary['files_per_dialogue']:.2f} per dialogue.")
    print(f"Total hours of questions = {summary['question_hours']:.2f}")
    print(f"Total hours of answers = {summary['answer_hours']:.2f}")
    print(f"(question + answer) = {summary['total_hours']:.2f}")
    print("###########################")


def fmt_count(value: float) -> str:
    return f"{int(round(value))}"


def fmt_hours(value: float) -> str:
    return f"{value:.2f}h"


def latex_row(label: str, values: list[str]) -> str:
    return f"    {label:<19} & " + " & ".join(values) + " \\\\ "


def print_latex_table(answers_root: Path, original_data_csv: Path, data_dir: Path) -> None:
    columns = [("improvised", "dev"), ("improvised", "test"), ("naturalistic", "dev"), ("naturalistic", "test")]
    summaries = [
        summarize_split_subset(answers_root / split / subset, original_data_csv, data_dir, split, subset)
        for subset, split in columns
    ]

    def values(key: str, formatter) -> list[str]:
        vals = [summary[key] for summary in summaries]
        return [formatter(value) for value in vals] + [formatter(sum(vals))]

    print(r"\\begin{tabular}{lccccc}")
    print(r"\\toprule")
    print("         & \\multicolumn{2}{c}{Improvised} & \\multicolumn{2}{c}{Naturalistic} & \\multirow{2}{*}{Total} \\\\ ")
    print("        & Dev & Test & Dev & Test &  \\\\ ")
    print(r"\\midrule")
    print(latex_row("Original sets", values("original_sets", fmt_count)))
    print(latex_row("Hours", values("original_hours", fmt_hours)))
    print(r"\\midrule")
    print(latex_row("Selected dialogues", values("selected_dialogues", fmt_count)))
    print(latex_row("Hours of questions", values("question_hours", fmt_hours)))
    print(latex_row("Hours of answers", values("answer_hours", fmt_hours)))
    print(latex_row("Total hours", values("total_hours", fmt_hours)))
    print(r"\\bottomrule")
    print(r"\\end{tabular}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_output_dir", type=Path, help="Path to one prepared answers split/subset directory.")
    parser.add_argument("--answers_root", type=Path, help="Path to the prepared answers root containing split/subset directories.")
    parser.add_argument("--original_data_csv", type=Path, required=True, help="Path to the CSV file containing the original data (e.g., seamless assets).")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to the root data directory containing the original audio files.")
    parser.add_argument("--split", type=str, help="Data split to analyze (e.g., 'test', 'dev').")
    parser.add_argument("--subset", type=str, help="Data subset to analyze (e.g., 'improvised', 'naturalistic').")
    parser.add_argument("--latex-table", action="store_true", help="Print a LaTeX table over improvised/naturalistic dev/test summaries.")
    args = parser.parse_args()

    if args.latex_table:
        if args.answers_root is None:
            parser.error("--latex-table requires --answers_root")
        print_latex_table(args.answers_root, args.original_data_csv, args.data_dir)
        return 0

    missing = [name for name in ["answers_output_dir", "split", "subset"] if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments for text summary: " + ", ".join(f"--{name}" for name in missing))

    summary = summarize_split_subset(args.answers_output_dir, args.original_data_csv, args.data_dir, args.split, args.subset)
    print_text_summary(summary, args.split, args.subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
