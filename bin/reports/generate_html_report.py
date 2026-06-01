#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Optional

import pandas as pd


SUBSETS = ["improvised", "naturalistic"]
EXCLUDED_REPORT_METRICS = {"question_end_time"}
DIALECT_LABELS = [
    "East Asia",
    "English",
    "Germanic",
    "Irish",
    "North America",
    "Northern Irish",
    "Oceania",
    "Other",
    "Romance",
    "Scottish",
    "Semitic",
    "Slavic",
    "South African",
    "Southeast Asia",
    "South Asia",
    "Welsh",
]


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def rel(path: Path, base: Path) -> str:
    try:
        return html.escape(str(path.relative_to(base)))
    except ValueError:
        return html.escape(str(path))


def fmt_float(value):
    if pd.isna(value):
        return ""
    try:
        value = float(value)
    except Exception:
        return value
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def table_html(frame: Optional[pd.DataFrame], columns=None, max_rows: Optional[int] = None) -> str:
    if frame is None or frame.empty:
        return '<p class="muted">No data available.</p>'
    view = frame.copy()
    if "metric" in view.columns:
        view = view[~view["metric"].astype(str).isin(EXCLUDED_REPORT_METRICS)]
    if columns is not None:
        view = view[[col for col in columns if col in view.columns]]
    if max_rows is not None:
        view = view.head(max_rows)
    for col in ["mean_diff", "std_diff", "p_value", "auroc", "accuracy"]:
        if col in view.columns:
            view[col] = view[col].map(fmt_float)
    if "n" in view.columns:
        view["n"] = pd.to_numeric(view["n"], errors="coerce").astype("Int64")
    return view.to_html(index=False, classes="metric-table", escape=True, border=0)


def read_report_text(report_dir: Path) -> str:
    path = report_dir / "report.txt"
    if not path.exists():
        return "report.txt not found. Run Stage 1 first."
    return path.read_text(encoding="utf-8")



def rows_to_html_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    thead = "<thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in header) + "</tr></thead>"
    tbody_rows = []
    for row in body:
        padded = row + [""] * max(0, len(header) - len(row))
        tbody_rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in padded[:len(header)]) + "</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f'<table class="metric-table">{thead}{tbody}</table>'


def render_stage1_report(report_text: str) -> str:
    blocks: list[str] = []
    table_rows: list[list[str]] = []
    paragraph_lines: list[str] = []
    lines = report_text.splitlines()
    idx = 0

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            blocks.append(rows_to_html_table(table_rows))
            table_rows = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = "<br>".join(html.escape(line) for line in paragraph_lines if line.strip())
            if text:
                blocks.append(f'<p>{text}</p>')
            paragraph_lines = []

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""

        if not stripped:
            flush_table()
            flush_paragraph()
            idx += 1
            continue

        if next_line and set(next_line) <= {"-", "="}:
            flush_table()
            flush_paragraph()
            level = "h2" if "=" in next_line else "h3"
            blocks.append(f'<{level}>{html.escape(stripped)}</{level}>')
            idx += 2
            continue

        if "\t" in line:
            flush_paragraph()
            table_rows.append(line.split("\t"))
        else:
            flush_table()
            paragraph_lines.append(line)
        idx += 1

    flush_table()
    flush_paragraph()
    return "\n".join(blocks)

def metric_rows(metrics: dict[str, Optional[pd.DataFrame]], section: str) -> pd.DataFrame:
    rows = []
    for subset, frame in metrics.items():
        if frame is None or frame.empty or "section" not in frame.columns:
            continue
        sub = frame[frame["section"] == section].copy()
        if "metric" in sub.columns:
            sub = sub[~sub["metric"].astype(str).isin(EXCLUDED_REPORT_METRICS)]
        if sub.empty:
            continue
        sub.insert(0, "subset", subset)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def language_id_summary_table(metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        frame = metrics.get(subset)
        row = {"subset": subset, "% Eng in models' answers": pd.NA, "% Eng in original answers": pd.NA}
        if frame is not None and not frame.empty and {"section", "metric", "mean_diff"}.issubset(frame.columns):
            language = frame[frame["section"] == "Language ID"]
            for _, source in language.iterrows():
                metric = str(source.get("metric", ""))
                if "model answers" in metric or "model's answers" in metric or "model answers" in metric:
                    row["% Eng in models' answers"] = source.get("mean_diff")
                elif "original answers" in metric or "original's answers" in metric:
                    row["% Eng in original answers"] = source.get("mean_diff")
        rows.append(row)
    return pd.DataFrame(rows)


def dialect_id_summary_table(metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        frame = metrics.get(subset)
        for label, prefix in [("Question", "Question dialect: "), ("Answer", "Answer dialect: ")]:
            row = {"Subset": subset, "Question/Answer": label}
            row.update({dialect: pd.NA for dialect in DIALECT_LABELS})
            if frame is not None and not frame.empty and {"section", "metric", "mean_diff"}.issubset(frame.columns):
                dialect_rows = frame[frame["section"] == "Dialect ID"]
                for _, source in dialect_rows.iterrows():
                    metric = str(source.get("metric", ""))
                    if not metric.startswith(prefix):
                        continue
                    dialect = metric[len(prefix):].removesuffix(" (%)")
                    if dialect in row:
                        row[dialect] = source.get("mean_diff")
            rows.append(row)
    return pd.DataFrame(rows)


def top_bottom_explainables(metrics: dict[str, Optional[pd.DataFrame]], subset: str, highest: bool) -> pd.DataFrame:
    frame = metrics.get(subset)
    if frame is None or frame.empty or "section" not in frame.columns or "auroc" not in frame.columns:
        return pd.DataFrame()
    sub = frame[frame["section"] == "Explainable Features"].copy()
    if "metric" in sub.columns:
        sub = sub[~sub["metric"].astype(str).isin(EXCLUDED_REPORT_METRICS)]
    sub["auroc"] = pd.to_numeric(sub["auroc"], errors="coerce")
    sub = sub.dropna(subset=["auroc"])
    return sub.sort_values("auroc", ascending=not highest).head(10)




def cluster_feature_rows(metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    frame = metric_rows(metrics, "Explainable Features")
    if frame.empty or "metric" not in frame.columns:
        return pd.DataFrame()
    metric = frame["metric"].astype(str)
    return frame[metric.str.startswith("corr_cluster_") | metric.str.startswith("general_explainable_feature_")].copy()

def img_tag(path: Path, report_dir: Path, alt: str, css_class: str = "figure") -> str:
    if not path.exists():
        return f'<p class="muted">Missing graph: {html.escape(str(path))}</p>'
    return f'<img class="{css_class}" src="{rel(path, report_dir)}" alt="{html.escape(alt)}" loading="lazy">'


def feature_gallery(report_dir: Path) -> str:
    graph_dir = report_dir / "graphs" / "feat_graphs"
    if not graph_dir.exists():
        return '<p class="muted">No per-feature graph directory found.</p>'
    images = [image for image in sorted(graph_dir.glob("*.png")) if image.stem not in EXCLUDED_REPORT_METRICS]
    if not images:
        return '<p class="muted">No per-feature graphs found.</p>'
    cards = []
    for image in images:
        title = html.escape(image.stem)
        cards.append(
            f'<figure class="thumb"><img src="{rel(image, report_dir)}" alt="{title}" loading="lazy"><figcaption>{title}</figcaption></figure>'
        )
    return '<details><summary>Per-feature histogram gallery</summary><div class="gallery">' + "\n".join(cards) + "</div></details>"



def basic_metric_gallery(report_dir: Path) -> str:
    graph_dir = report_dir / "graphs" / "basic_metric_graphs"
    if not graph_dir.exists():
        return '<p class="muted">No basic-metric graph directory found.</p>'
    images = []
    for image in sorted(graph_dir.glob("*.png")):
        stem = image.stem
        if stem == "interruptions" or stem.startswith("WER_") or stem.startswith("CER_"):
            continue
        images.append(image)
    if not images:
        return '<p class="muted">No basic-metric graphs found.</p>'
    cards = []
    for image in images:
        title = html.escape(image.stem)
        cards.append(
            f'<figure class="thumb"><img src="{rel(image, report_dir)}" alt="{title}" loading="lazy"></figure>'
        )
    return '<details open><summary>Basic metric histogram gallery</summary><div class="gallery">' + "\n".join(cards) + "</div></details>"

def summary_cards(metrics: dict[str, Optional[pd.DataFrame]]) -> str:
    cards = []
    for subset, frame in metrics.items():
        if frame is None or frame.empty:
            continue
        emo = frame[frame["section"] == "Emotional Naturalness"] if "section" in frame else pd.DataFrame()
        basic = frame[frame["section"] == "Basic Metrics"] if "section" in frame else pd.DataFrame()
        language = frame[frame["section"] == "Language ID"] if "section" in frame else pd.DataFrame()
        dialect = frame[frame["section"] == "Dialect ID"] if "section" in frame else pd.DataFrame()
        explain = frame[frame["section"] == "Explainable Features"] if "section" in frame else pd.DataFrame()
        stances = frame[frame["section"] == "STANCEs"] if "section" in frame else pd.DataFrame()
        bits = [f'<h3>{html.escape(subset.title())}</h3>']
        if not emo.empty:
            row = emo.iloc[0]
            bits.append(f'<p><strong>Emotional naturalness diff:</strong> {fmt_float(row.get("mean_diff"))} (p={fmt_float(row.get("p_value"))})</p>')
        if not basic.empty:
            bits.append(f'<p><strong>Basic metrics:</strong> {len(basic)} metrics summarized</p>')
        if not language.empty:
            language_values = []
            for _, language_row in language.iterrows():
                metric_name = str(language_row.get("metric", ""))
                percent = fmt_float(language_row.get("mean_diff"))
                if percent:
                    language_values.append(f'{html.escape(metric_name)}: {percent}%')
            if language_values:
                bits.append(f'<p><strong>Language ID:</strong> {"; ".join(language_values)}</p>')
        if not dialect.empty:
            bits.append(f'<p><strong>Dialect ID:</strong> {len(dialect)} dialect percentage rows</p>')
        if not stances.empty:
            stance_questions = stances[stances["metric"].astype(str).str.match(r"^Q\d+$")] if "metric" in stances else stances
            mean_stance = pd.to_numeric(stance_questions["mean_diff"], errors="coerce").mean()
            bits.append(f'<p><strong>Mean STANCE diff:</strong> {fmt_float(mean_stance)} across {len(stance_questions)} stances</p>')
        if not explain.empty:
            best_idx = pd.to_numeric(explain["auroc"], errors="coerce").idxmax()
            best = explain.loc[best_idx]
            bits.append(f'<p><strong>Best explainable AUROC:</strong> {html.escape(str(best.get("metric")))} ({fmt_float(best.get("auroc"))})</p>')
        cards.append('<section class="card">' + "\n".join(bits) + '</section>')
    return '<div class="cards">' + "\n".join(cards) + '</div>'


def build_html(args, metrics: dict[str, Optional[pd.DataFrame]], report_text: str) -> str:
    report_dir = args.report_dir
    emo_table = metric_rows(metrics, "Emotional Naturalness")
    basic_table = metric_rows(metrics, "Basic Metrics")
    language_table = language_id_summary_table(metrics)
    dialect_table = dialect_id_summary_table(metrics)
    stance_table = metric_rows(metrics, "STANCEs")
    cluster_feature_table = cluster_feature_rows(metrics)

    explain_sections = []
    for subset in SUBSETS:
        explain_sections.append(f'<h3>{html.escape(subset.title())}</h3>')
        explain_sections.append('<h4>Highest AUROC Features</h4>')
        explain_sections.append(table_html(top_bottom_explainables(metrics, subset, True), columns=["metric", "mean_diff", "std_diff", "p_value", "n", "auroc", "accuracy"]))
        explain_sections.append('<h4>Lowest AUROC Features</h4>')
        explain_sections.append(table_html(top_bottom_explainables(metrics, subset, False), columns=["metric", "mean_diff", "std_diff", "p_value", "n", "auroc", "accuracy"]))

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPEARBench Detailed Report - {html.escape(args.model)}</title>
  <style>
    :root {{ --ink:#17212b; --muted:#5b6773; --line:#d8dee6; --bg:#f7f9fb; --card:#ffffff; --accent:#225ea8; }}
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:var(--ink); background:var(--bg); line-height:1.5; }}
    header {{ padding:32px 44px; background:#0f2438; color:white; }}
    header h1 {{ margin:0 0 8px 0; font-size:30px; }}
    header p {{ margin:0; color:#dbe7f3; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 24px 56px; }}
    section {{ margin:0 0 32px; }}
    h2 {{ margin-top:0; border-bottom:2px solid var(--line); padding-bottom:8px; }}
    h3 {{ margin-bottom:8px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; margin:18px 0; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }}
    .metric-table {{ width:100%; border-collapse:collapse; font-size:14px; background:white; }}
    .metric-table th, .metric-table td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }}
    .metric-table th {{ background:#eef3f8; font-weight:700; }}
    .figure {{ width:100%; max-width:1120px; display:block; margin:14px auto 24px; border:1px solid var(--line); border-radius:6px; background:white; }}
    .muted {{ color:var(--muted); }}
    details {{ margin-top:16px; }}
    summary {{ cursor:pointer; color:var(--accent); font-weight:700; margin-bottom:12px; }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    .thumb {{ margin:0; background:white; border:1px solid var(--line); border-radius:6px; padding:8px; }}
    .thumb img {{ width:100%; display:block; }}
    .thumb figcaption {{ font-size:12px; color:var(--muted); padding-top:6px; word-break:break-word; }}
  </style>
</head>
<body>
<header>
  <h1>SPEARBench Detailed Report</h1>
  <p>Model: {html.escape(args.model)} | Protocol: {html.escape(args.protocol)}</p>
</header>
<main>
  <section>
    <h2>Executive Summary</h2>
    <p>This report combines the simplified text report, Stage 1 metric tables, and Stage 2 figures into a browsable HTML document.</p>
    {summary_cards(metrics)}
  </section>


  <section>
    <h2>Basic Metrics</h2>
    <p>Basic metrics come from script 10 and include ASR-specific CER/WER columns, UTMOS, latency, interruption counts, and interruption overlap duration when available. If original base metrics were not generated, the table reports model means and standard deviations.</p>
    {table_html(basic_table, columns=["subset", "metric", "mean_diff", "std_diff", "p_value", "n", "detail"])}
    {img_tag(report_dir / "graphs" / "basic_metrics.png", report_dir, "Basic metric violin plots")}
    {basic_metric_gallery(report_dir)}
  </section>

  <section>
    <h2>Language ID</h2>
    <p>Language ID reports the percentage of answer audio rows whose detected language is English (<code>eng</code>) in <code>language_id.csv</code>.</p>
    {table_html(language_table, columns=["subset", "% Eng in models' answers", "% Eng in original answers"])}
  </section>

  <section>
    <h2>Dialect ID</h2>
    <p>Dialect ID reports the percentage of question and answer rows assigned to each VoxLect dialect class. The confusion matrix counts question-to-answer dialect changes; the box plot compares the full question and answer score vectors for each class.</p>
    {table_html(dialect_table, columns=["Subset", "Question/Answer", *DIALECT_LABELS])}
    {img_tag(report_dir / "graphs" / "dialect_confusion.png", report_dir, "Dialect question-to-answer confusion matrix")}
    {img_tag(report_dir / "graphs" / "dialect_scores.png", report_dir, "Dialect score distributions for question and answer fields")}
  </section>

  <section>
    <h2>Emotional Naturalness</h2>
    <p>Naturalness is summarized as the paired test-set logit difference between model and original utterances. Positive values mean higher model logits than original logits.</p>
    {table_html(emo_table, columns=["subset", "metric", "mean_diff", "std_diff", "p_value", "n"])}
    {img_tag(report_dir / "graphs" / "emo_naturalness.png", report_dir, "Emotional naturalness distributions")}
  </section>

  <section>
    <h2>STANCEs</h2>
    <p>STANCE metrics compare LLM and original scores for each STANCE question. Naturalistic STANCEs are omitted because they are not computed in this benchmark pipeline.</p>
    {table_html(stance_table, columns=["subset", "metric", "mean_diff", "std_diff", "p_value", "n", "detail"])}
    {img_tag(report_dir / "graphs" / "stances.png", report_dir, "STANCE score distributions")}
  </section>

  <section>
    <h2>Explainable Features</h2>
    <p>Explainable feature rows report model-minus-original mean differences, p-values, and the Stage 41 classification AUROC/accuracy. Correlation-cluster rows come from script 41 PCA cluster features.</p>
    <h3>Correlation Cluster Features</h3>
    {table_html(cluster_feature_table, columns=["subset", "metric", "mean_diff", "std_diff", "p_value", "n", "detail"])}
    {img_tag(report_dir / "graphs" / "cluster_explainables.png", report_dir, "Correlation cluster explainable feature violins")}
    {img_tag(report_dir / "graphs" / "general_explainable.png", report_dir, "General explainable feature histogram")}
    {''.join(explain_sections)}
    {img_tag(report_dir / "graphs" / "explainables.png", report_dir, "Explainable feature violin plots")}
    {feature_gallery(report_dir)}
  </section>
</main>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--ignore-features", nargs="*", default=[])
    args = parser.parse_args()
    EXCLUDED_REPORT_METRICS.update(args.ignore_features)

    metrics = {subset: safe_read_csv(args.report_dir / f"metrics-{subset}.csv") for subset in SUBSETS}
    report_text = read_report_text(args.report_dir)
    html_text = build_html(args, metrics, report_text)
    output_path = args.report_dir / "detailed_report.html"
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
