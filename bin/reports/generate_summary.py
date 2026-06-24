#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from html import unescape
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
MAX_REPORT_CHARS = 120_000


def strip_graphs_and_html(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</(h[1-6]|p|tr|li|section|div|table)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def read_long_report_without_graphs(report_dir: Path) -> str:
    html_path = report_dir / "detailed_report.html"
    text_path = report_dir / "report.txt"
    if html_path.exists():
        return strip_graphs_and_html(html_path.read_text(encoding="utf-8", errors="replace"))
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(f"Could not find {html_path} or {text_path}")


def trim_report(text: str) -> str:
    if len(text) <= MAX_REPORT_CHARS:
        return text
    head = text[: MAX_REPORT_CHARS // 2]
    tail = text[-MAX_REPORT_CHARS // 2 :]
    return f"{head}\n\n[... report truncated for summarization ...]\n\n{tail}"


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text.strip()
    payload = response.model_dump() if hasattr(response, "model_dump") else response
    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def request_summary(api_key: str, organization: str | None, model: str, protocol: str, report_model: str, report_text: str) -> str:
    prompt = f"""You are summarizing a SPEARBench model evaluation report.

Write one short, self-contained paragraph, 5 to 8 sentences maximum, describing the model performance for a technical reader. Mention the main strengths, weaknesses, and notable tradeoffs across intelligibility/interruption metrics, language/dialect behavior, emotion/stance behavior, and explainable speech features when those signals are present. Do not mention graphs or HTML. Do not invent numbers; use only the report content.

Protocol: {protocol}
Model being evaluated: {report_model}

Report text without graphs:
{trim_report(report_text)}
"""
    client_kwargs: dict[str, str] = {"api_key": api_key}
    if organization:
        client_kwargs["organization"] = organization
    client = OpenAI(**client_kwargs)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        max_output_tokens=700,
    )
    text = extract_response_text(response)
    if not text:
        raise RuntimeError("OpenAI API response did not contain summary text")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Create summary.txt for a SPEARBench detailed report.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True, help="Model being evaluated in the report.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--summary-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--organization", default="")
    args = parser.parse_args()

    report_text = read_long_report_without_graphs(args.report_dir)
    summary = request_summary(
        args.api_key,
        args.organization or None,
        args.summary_model,
        args.protocol,
        args.model,
        report_text,
    )
    output_path = args.report_dir / "summary.txt"
    output_path.write_text(summary.strip() + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
