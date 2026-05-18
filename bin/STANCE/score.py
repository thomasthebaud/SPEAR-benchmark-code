#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TAU0_DEFAULT = 0.45
TAU2_DEFAULT = 0.75
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text.strip()


def str_to_bool(value: Any) -> bool:
    return safe_str(value).lower() in {"1", "true", "t", "yes", "y"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = safe_str(text)
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def coerce_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        return [safe_str(item) for item in value if safe_str(item)]
    if safe_str(value):
        return [safe_str(value)]
    return []


def map_choice_probability_to_score(choice: str, probability: float) -> int:
    choice = safe_str(choice).upper()
    if choice not in {"P", "N"}:
        raise ValueError(f"Invalid choice: {choice!r}")
    probability = max(0.0, min(1.0, float(probability)))
    sign = 1 if choice == "P" else -1
    if probability < TAU0_DEFAULT:
        return 0
    if probability >= TAU2_DEFAULT:
        return 2 * sign
    return sign


def audio_part(audio_path: Path) -> dict[str, Any]:
    fmt = "mp3" if audio_path.suffix.lower() == ".mp3" else "wav"
    with audio_path.open("rb") as infile:
        audio_b64 = base64.b64encode(infile.read()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}}


def build_prompt(row: dict[str, str]) -> str:
    target_bits = [
        f"TARGET speaker id: {safe_str(row.get('target_speaker')) or 'unknown'}",
        f"TARGET role: {safe_str(row.get('target_role')) or 'unknown'}",
        f"TARGET category: {safe_str(row.get('target_category')) or 'unknown'}",
    ]
    question_end_time = safe_str(row.get("question_end_time"))
    if question_end_time:
        target_bits.append(
            f"The original dialogue/question ends at {question_end_time} seconds; evaluate the answer/response after that boundary when present."
        )

    return (
        "Judge the TARGET speaker's stance/tone/style in the audio.\n"
        "Use any dialogue context only to interpret the TARGET response; do not score the other speaker.\n\n"
        + "\n".join(target_bits)
        + "\n\n"
        f"Question: {safe_str(row.get('stance_question'))}\n\n"
        f"Definition P:\n{safe_str(row.get('positive_definition'))}\n\n"
        f"Definition N:\n{safe_str(row.get('negative_definition'))}\n\n"
        f"Relationship/context: {safe_str(row.get('relationship_detail'))}\n"
        f"Participant A prompt: {safe_str(row.get('participant_a_prompt_text'))}\n"
        f"Participant B prompt: {safe_str(row.get('participant_b_prompt_text'))}\n"
        f"Transcript: {safe_str(row.get('transcript'))}\n\n"
        "Return ONLY JSON with keys 'choice', 'probability', and 'evidence'. "
        "'choice' must be 'P' or 'N'. 'probability' must be 0.0 to 1.0. "
        "'evidence' must be 2 to 4 short strings."
    )


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float,
        max_retries: int,
        organization: str = "",
        project: str = "",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.url += "/chat/completions"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.organization = organization
        self.project = project

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers

    def complete(self, audio_path: Path, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "modalities": ["text"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "stance_judge_output",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "choice": {"type": "string", "enum": ["P", "N"]},
                            "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 4,
                            },
                        },
                        "required": ["choice", "probability", "evidence"],
                    },
                    "strict": True,
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful dialogue analyst. "
                        "You judge audio prosody, wording, and interactional stance."
                    ),
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}, audio_part(audio_path)]},
            ],
        }

        last_error = ""
        for attempt in range(self.max_retries + 1):
            data = json.dumps(payload).encode("utf-8")
            request = Request(self.url, data=data, headers=self.headers(), method="POST")
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                return safe_str(content)
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body}"
                if exc.code == 400 and "max_completion_tokens" in body and "max_completion_tokens" in payload:
                    payload["max_tokens"] = payload.pop("max_completion_tokens")
                    continue
                if exc.code == 400 and "response_format" in body and "response_format" in payload:
                    payload.pop("response_format", None)
                    continue
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise RuntimeError(last_error) from exc
            except URLError as exc:
                last_error = f"URL error: {exc}"

            if attempt < self.max_retries:
                time.sleep(min(30.0, 1.5 * (2**attempt)))
        raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def score_row(row: dict[str, str], client: OpenAIChatClient, temperature: float, max_tokens: int) -> dict[str, Any]:
    audio_path = Path(safe_str(row.get("audio_path")))
    if not audio_path.exists():
        return {
            "stance_choice": "",
            "stance_probability": "",
            "stance_score": "",
            "stance_evidence": "[]",
            "stance_raw_response": "",
            "stance_error": f"missing audio: {audio_path}",
        }

    raw = client.complete(audio_path=audio_path, prompt=build_prompt(row), temperature=temperature, max_tokens=max_tokens)
    obj = extract_json_object(raw)
    if obj is None:
        return {
            "stance_choice": "",
            "stance_probability": "",
            "stance_score": "",
            "stance_evidence": "[]",
            "stance_raw_response": raw,
            "stance_error": "invalid_json",
        }

    choice = safe_str(obj.get("choice")).upper()
    probability = float(obj.get("probability"))
    score = map_choice_probability_to_score(choice, probability)
    evidence = coerce_evidence(obj.get("evidence"))[:4]
    return {
        "stance_choice": choice,
        "stance_probability": probability,
        "stance_score": score,
        "stance_evidence": json.dumps(evidence, ensure_ascii=False),
        "stance_raw_response": raw,
        "stance_error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=None, help="Accepted for shell-script compatibility; not used.")
    parser.add_argument("--questions-csv", type=Path, required=True)
    parser.add_argument("--outputs_dir", "--outputs-dir", dest="outputs_dir", type=Path, required=True)
    parser.add_argument("--eval_model", "--eval-model", dest="eval_model", default=os.environ.get("OPENAI_MODEL", "gpt-audio-2025-08-28"))
    parser.add_argument("--metrics-name", default=None)
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-org", default=os.environ.get("OPENAI_ORG", ""))
    parser.add_argument("--openai-project", default=os.environ.get("OPENAI_PROJECT", ""))
    parser.add_argument("--openai-timeout-s", type=float, default=float(os.environ.get("OPENAI_TIMEOUT_S", 120)))
    parser.add_argument("--openai-max-retries", type=int, default=int(os.environ.get("OPENAI_MAX_RETRIES", 6)))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    rows = read_csv_rows(args.questions_csv)
    valid_rows = [row for row in rows if str_to_bool(row.get("can_use_for_stance"))]
    skipped_count = len(rows) - len(valid_rows)

    if skipped_count:
        print(f"Skipping {skipped_count}/{len(rows)} rows with can_use_for_stance=false.", flush=True)
    if not valid_rows:
        print(f"No valid STANCE rows found in {args.questions_csv}; writing an empty metrics CSV.", flush=True)
        output_name = args.metrics_name or f"stance_metrics_Q{safe_str(rows[0].get('question_index')) if rows else 'NA'}.csv"
        output_path = args.outputs_dir / output_name
        original_fields = list(rows[0].keys()) if rows else []
        metric_fields = [
            "stance_choice",
            "stance_probability",
            "stance_score",
            "stance_evidence",
            "stance_raw_response",
            "stance_error",
        ]
        write_csv_rows(output_path, [], original_fields + metric_fields)
        print(f"Wrote STANCE metrics to {output_path}")
        return

    if not args.openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Export it or pass --openai-api-key.")

    client = OpenAIChatClient(
        api_key=args.openai_api_key,
        model=args.eval_model,
        base_url=args.openai_base_url,
        timeout_s=args.openai_timeout_s,
        max_retries=args.openai_max_retries,
        organization=args.openai_org,
        project=args.openai_project,
    )

    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(valid_rows):
        print(f"[{idx + 1}/{len(valid_rows)}] scoring row_idx={safe_str(row.get('row_idx'))}", flush=True)
        result = score_row(row, client, args.temperature, args.max_tokens)
        out = dict(row)
        out.update(result)
        scored_rows.append(out)

    output_name = args.metrics_name or f"stance_metrics_Q{safe_str(valid_rows[0].get('question_index'))}.csv"
    output_path = args.outputs_dir / output_name
    original_fields = list(valid_rows[0].keys())
    metric_fields = [
        "stance_choice",
        "stance_probability",
        "stance_score",
        "stance_evidence",
        "stance_raw_response",
        "stance_error",
    ]
    write_csv_rows(output_path, scored_rows, original_fields + metric_fields)
    print(f"Wrote STANCE metrics to {output_path}")


if __name__ == "__main__":
    main()
