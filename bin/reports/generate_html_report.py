#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
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
    numeric_display_cols = [
        col
        for col in view.columns
        if col in {"mean_diff", "std_diff", "p_value", "auroc", "accuracy"}
        or col.startswith("model_mean_")
        or col.startswith("original_mean_")
        or col.startswith("diff_")
    ]
    for col in numeric_display_cols:
        non_null = view[col].dropna()
        if not non_null.empty and non_null.map(lambda value: isinstance(value, str)).all():
            continue
        view[col] = view[col].map(fmt_float)
    if "n" in view.columns:
        view["n"] = pd.to_numeric(view["n"], errors="coerce").astype("Int64")
    return view.to_html(index=False, classes="metric-table", escape=True, border=0)


def read_report_text(report_dir: Path) -> str:
    path = report_dir / "report.txt"
    if not path.exists():
        return "report.txt not found. Run Stage 1 first."
    return path.read_text(encoding="utf-8")


def read_summary_text(report_dir: Path) -> str:
    path = report_dir / "summary.txt"
    if not path.exists():
        return "missing summary"
    text = path.read_text(encoding="utf-8").strip()
    return text or "missing summary"



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


def extract_named_sections(report_text: str, names: set[str]) -> str:
    lines = report_text.splitlines()
    selected: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if line.strip() in names and next_line.strip() and set(next_line.strip()) <= {"-", "="}:
            start = idx
            idx += 2
            while idx < len(lines):
                candidate_next = lines[idx + 1] if idx + 1 < len(lines) else ""
                if lines[idx].strip() and candidate_next.strip() and set(candidate_next.strip()) <= {"-", "="}:
                    break
                idx += 1
            selected.extend(lines[start:idx])
            selected.append("")
            continue
        idx += 1
    return "\n".join(selected).strip()


def selected_stage1_html(report_text: str, names: set[str]) -> str:
    selected = extract_named_sections(report_text, names)
    if not selected:
        return '<p class="muted">No setup text available.</p>'
    return render_stage1_report(selected)


def section_body_lines(report_text: str, name: str) -> list[str]:
    lines = report_text.splitlines()
    for idx, line in enumerate(lines):
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if line.strip() == name and next_line.strip() and set(next_line.strip()) <= {"-", "="}:
            body: list[str] = []
            scan = idx + 2
            while scan < len(lines):
                candidate_next = lines[scan + 1] if scan + 1 < len(lines) else ""
                if lines[scan].strip() and candidate_next.strip() and set(candidate_next.strip()) <= {"-", "="}:
                    break
                if lines[scan].strip():
                    body.append(lines[scan].rstrip())
                scan += 1
            return body
    return []


def parse_aligned_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if not line.strip():
            continue
        cells = [cell.strip() for cell in re.split(r"\s{2,}", line.strip())]
        if len(cells) >= 2:
            rows.append(cells)
    return rows


def report_section_table(report_text: str, name: str) -> str:
    rows = parse_aligned_table(section_body_lines(report_text, name))
    if not rows:
        return '<p class="muted">No data available.</p>'
    return rows_to_html_table(rows)


def data_models_setup_grid(report_text: str) -> str:
    data = (
        '<section class="setup-panel setup-panel-data">'
        '<h3>Data</h3>'
        f'{report_section_table(report_text, "Data")}'
        '</section>'
    )
    lower_cards = []
    for name in ["Models", "Setup"]:
        lower_cards.append(
            '<section class="setup-panel">'
            f'<h3>{html.escape(name)}</h3>'
            f'{report_section_table(report_text, name)}'
            '</section>'
        )
    return data + '<div class="setup-half-grid">' + "\n".join(lower_cards) + '</div>'


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
        row = {
            "subset": subset,
            "% Eng in models' answers": pd.NA,
            "% Eng in original answers": pd.NA,
            "second most spoken language in models' answers": pd.NA,
            "percentage 2nd language in models' answers": pd.NA,
            "second most spoken language in original answers": pd.NA,
            "percentage 2nd language in original answers": pd.NA,
        }
        if frame is not None and not frame.empty and {"section", "metric", "mean_diff"}.issubset(frame.columns):
            language = frame[frame["section"].isin(["Language ID", "Language and Dialect ID"])]
            for _, source in language.iterrows():
                metric = str(source.get("metric", ""))
                detail = str(source.get("detail", ""))
                second_language = ""
                if "language=" in detail:
                    second_language = detail.split("language=", 1)[1].split(";", 1)[0]
                if metric == "English language detected in model answers (%)":
                    row["% Eng in models' answers"] = source.get("mean_diff")
                elif metric == "English language detected in original answers (%)":
                    row["% Eng in original answers"] = source.get("mean_diff")
                elif metric == "Second most spoken language in model answers":
                    row["second most spoken language in models' answers"] = second_language
                    row["percentage 2nd language in models' answers"] = source.get("mean_diff")
                elif metric == "Second most spoken language in original answers":
                    row["second most spoken language in original answers"] = second_language
                    row["percentage 2nd language in original answers"] = source.get("mean_diff")
        rows.append(row)

    table = pd.DataFrame(rows)
    for eng_col, second_cols in [
        ("% Eng in models' answers", ["second most spoken language in models' answers", "percentage 2nd language in models' answers"]),
        ("% Eng in original answers", ["second most spoken language in original answers", "percentage 2nd language in original answers"]),
    ]:
        values = pd.to_numeric(table[eng_col], errors="coerce") if eng_col in table else pd.Series(dtype="float64")
        if not values.dropna().empty and (values.dropna() >= 100.0).all():
            table = table.drop(columns=[col for col in second_cols if col in table])
    return table


def dialect_id_summary_table(metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        frame = metrics.get(subset)
        for label, prefix in [("Question", "Question dialect: "), ("Answer", "Answer dialect: ")]:
            row = {"Subset": subset, "Question/Answer": label}
            row.update({dialect: pd.NA for dialect in DIALECT_LABELS})
            if frame is not None and not frame.empty and {"section", "metric", "mean_diff"}.issubset(frame.columns):
                dialect_rows = frame[frame["section"].isin(["Dialect ID", "Language and Dialect ID"])]
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


def split_basic_metric(metric: str) -> tuple[str, str]:
    metric = str(metric)
    for prefix in ["CER_", "WER_"]:
        if metric.startswith(prefix):
            return prefix[:-1], metric[len(prefix):]
    return metric, ""


def format_basic_table_value(metric: str, value, *, diff: bool = False):
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, str) and "/" in value:
        return value
    try:
        number = float(value)
    except Exception:
        return value
    if metric in {"CER", "WER", "average number of interruptions per dialogues"}:
        return f"{number:.2f}"
    if metric == "interrupted time (s)":
        return f"{number:.3f}"
    if metric == "number of dialogues with interruption" and diff:
        return f"{int(round(number))}"
    return f"{number:.4f}"


def basic_display_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Metric",
        "System",
        "model_mean_naturalistic",
        "original_mean_naturalistic",
        "model_mean_improvised",
        "original_mean_improvised",
        "diff_improvised",
        "diff_naturalistic",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows = {}
    order = []
    for _, source in frame.iterrows():
        metric, system = split_basic_metric(source.get("metric", ""))
        key = (metric, system)
        if key not in rows:
            rows[key] = {column: pd.NA for column in columns}
            rows[key]["Metric"] = metric
            rows[key]["System"] = system
            order.append(key)
        subset = str(source.get("subset", "")).strip().lower()
        if subset not in {"improvised", "naturalistic"}:
            continue
        rows[key][f"model_mean_{subset}"] = format_basic_table_value(metric, source.get("model_value"))
        rows[key][f"original_mean_{subset}"] = format_basic_table_value(metric, source.get("original_value"))
        rows[key][f"diff_{subset}"] = format_basic_table_value(metric, source.get("mean_diff"), diff=True)

    return pd.DataFrame([rows[key] for key in order], columns=columns)


def stance_description_table(metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    frame = metric_rows(metrics, "STANCE Descriptions")
    if frame.empty:
        return frame
    out = frame.copy()
    out["stance"] = out["metric"].astype(str).str.replace(" stance description", "", regex=False)
    out = out.rename(columns={"model_value": "positive_original_%", "original_value": "negative_original_%", "detail": "definition"})
    return out[[col for col in ["stance", "positive_original_%", "negative_original_%", "definition"] if col in out.columns]]


def stance_result_table(metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    frame = metric_rows(metrics, "STANCE Results")
    if frame.empty:
        return frame
    out = frame.copy()
    out["dataset"] = out["metric"].astype(str).str.replace("STANCE results: ", "", regex=False)
    out = out.rename(columns={"mean_diff": "STANCE same sign (%)", "model_value": "More positive (%)", "original_value": "More negative (%)"})
    return out[[col for col in ["dataset", "STANCE same sign (%)", "More positive (%)", "More negative (%)"] if col in out.columns]]

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
        if stem.startswith("WER_") or stem.startswith("CER_"):
            continue
        if stem in {"interrupted", "interruption_segments", "interruptions"}:
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
        language = frame[frame["section"].isin(["Language ID", "Language and Dialect ID"])] if "section" in frame else pd.DataFrame()
        dialect = frame[frame["section"].isin(["Dialect ID", "Language and Dialect ID"])] if "section" in frame else pd.DataFrame()
        explain = frame[frame["section"] == "Explainable Features"] if "section" in frame else pd.DataFrame()
        stances = frame[frame["section"] == "STANCEs"] if "section" in frame else pd.DataFrame()
        bits = [f'<h3>{html.escape(subset.title())}</h3>']
        if not emo.empty:
            row = emo.iloc[0]
            bits.append(f'<p><strong>Emotional naturalness normalized-logit diff:</strong> {fmt_float(row.get("mean_diff"))} (p={fmt_float(row.get("p_value"))})</p>')
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
        cards.append('<section class="card">' + "\n".join(bits) + '</section>')
    return '<div class="cards">' + "\n".join(cards) + '</div>'


def dialectal_metrics_table(metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    wanted = {"Dialectal entrainment", "Dialectal variance"}
    for subset in SUBSETS:
        frame = metrics.get(subset)
        if frame is None or frame.empty or not {"section", "metric"}.issubset(frame.columns):
            continue
        sub = frame[(frame["section"] == "Language and Dialect ID") & frame["metric"].astype(str).isin(wanted)].copy()
        if sub.empty:
            continue
        sub.insert(0, "subset", subset)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_html(args, metrics: dict[str, Optional[pd.DataFrame]], report_text: str) -> str:
    report_dir = args.report_dir
    emo_table = metric_rows(metrics, "Emotional Naturalness")
    basic_table = metric_rows(metrics, "Intelligibility and Interruption Metrics")
    language_table = language_id_summary_table(metrics)
    dialect_table = dialect_id_summary_table(metrics)
    stance_desc_table = metric_rows(metrics, "STANCE Descriptions")
    stance_results_table = metric_rows(metrics, "STANCE Results")
    dialectal_table = dialectal_metrics_table(metrics)
    summary_text = read_summary_text(report_dir)

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
    .setup-half-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; align-items:start; margin-top:16px; }}
    .setup-panel {{ min-width:0; overflow-x:auto; margin:0; }}
    .setup-panel-data {{ width:100%; }}
    .setup-panel h3 {{ margin-top:0; }}
    .metric-table {{ width:100%; border-collapse:collapse; font-size:14px; background:white; }}
    .metric-table th, .metric-table td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }}
    .metric-table th {{ background:#eef3f8; font-weight:700; }}
    .figure {{ width:100%; max-width:1120px; display:block; margin:14px auto 24px; border:1px solid var(--line); border-radius:6px; background:white; }}
    .muted {{ color:var(--muted); }}
    .summary-paragraph {{ background:white; border:1px solid var(--line); border-radius:8px; padding:16px; white-space:pre-wrap; }}
    details {{ margin-top:16px; }}
    summary {{ cursor:pointer; color:var(--accent); font-weight:700; margin-bottom:12px; }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    .thumb {{ margin:0; background:white; border:1px solid var(--line); border-radius:6px; padding:8px; }}
    .thumb img {{ width:100%; display:block; }}
    .thumb figcaption {{ font-size:12px; color:var(--muted); padding-top:6px; word-break:break-word; }}
    @media (max-width: 900px) {{ .setup-half-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>SPEARBench Detailed Report</h1>
  <p>Model: {html.escape(args.model)} | Protocol: {html.escape(args.protocol)}</p>
</header>
<main>
  <section>
    <h2>Summary paragraph</h2>
    <p class="summary-paragraph">{html.escape(summary_text)}</p>
  </section>

  <section>
    <h2>Data, Models, and Setup</h2>
    {data_models_setup_grid(report_text)}
  </section>

  <section>
    <h2>Intelligibility and Interruption Metrics</h2>
    {table_html(basic_display_table(basic_table), columns=["Metric", "System", "model_mean_naturalistic", "original_mean_naturalistic", "model_mean_improvised", "original_mean_improvised"])}
    {img_tag(report_dir / "graphs" / "basic_metrics.png", report_dir, "Intelligibility and interruption metric histograms with KDE")}
    {basic_metric_gallery(report_dir)}
  </section>

  <section>
    <h2>Language and Dialect ID</h2>
    {table_html(language_table, columns=["subset", "% Eng in models' answers", "% Eng in original answers", "second most spoken language in models' answers", "percentage 2nd language in models' answers", "second most spoken language in original answers", "percentage 2nd language in original answers"])}
    {table_html(dialect_table, columns=["Subset", "Question/Answer", *DIALECT_LABELS])}
    <h3>Dialectal Metrics</h3>
    {table_html(dialectal_table, columns=["subset", "metric", "mean_diff", "n", "detail"])}
    {img_tag(report_dir / "graphs" / "dialect_confusion.png", report_dir, "Dialect question-to-answer confusion matrix")}
    {img_tag(report_dir / "graphs" / "dialect_scores.png", report_dir, "Dialect score spider profiles for question and answer fields")}
  </section>

  <section>
    <h2>Emotional Naturalness</h2>
    {img_tag(report_dir / "graphs" / "emo_naturalness.png", report_dir, "Emotional naturalness distributions")}
    <p>The relationship violin plot breaks naturalistic emotional naturalness logits down by relationship label.</p>
    {img_tag(report_dir / "graphs" / "emo_naturalness_by_relationship.png", report_dir, "Naturalistic emotional naturalness violins by relationship")}
    <p>The emotion scatter plot compares full-question and full-answer Arousal, Dominance, and Valence scores normalized to [-1, 1]. Dotted lines show per-dataset correlations with rho labels.</p>
    {img_tag(report_dir / "graphs" / "emotion_scatter.png", report_dir, "Question-answer Arousal, Dominance, and Valence scatter plots")}
  </section>

  <section>
    <h2>STANCE</h2>
    <h3>Stance Descriptions</h3>
    {table_html(stance_description_table(metrics), columns=["stance", "positive_original_%", "negative_original_%", "definition"])}
    <h3>Results</h3>
    {table_html(stance_result_table(metrics), columns=["dataset", "STANCE same sign (%)", "More positive (%)", "More negative (%)"])}
    {img_tag(report_dir / "graphs" / "stances.png", report_dir, "STANCE score distributions")}
  </section>

  <section>
    <h2>Explainable Features</h2>
    <p>Explainable feature rows report model-minus-original mean differences and p-values for the reported f0, duration, and voiced-ratio features.</p>
    {img_tag(report_dir / "graphs" / "f0_question_answer_boxplot.png", report_dir, "Questions, original answers, and model answers f0 profile box plot")}
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
