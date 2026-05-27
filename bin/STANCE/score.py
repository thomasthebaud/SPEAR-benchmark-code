#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import tempfile
import time
import wave
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from tqdm import tqdm

import audioop


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


def load_wav_mono(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path), "rb") as infile:
        channels = infile.getnchannels()
        sample_width = infile.getsampwidth()
        sample_rate = infile.getframerate()
        frames = infile.readframes(infile.getnframes())

    if sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}")
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
    elif channels != 1:
        raise ValueError(f"Unsupported WAV channel count {channels} for {path}")
    return frames, int(sample_rate), int(sample_width)


def convert_wav_audio(audio: bytes, sr_in: int, sample_width_in: int, sr_out: int, sample_width_out: int) -> bytes:
    if sample_width_in != sample_width_out:
        audio = audioop.lin2lin(audio, sample_width_in, sample_width_out)
    if sr_in != sr_out and audio:
        audio, _state = audioop.ratecv(audio, sample_width_out, 1, sr_in, sr_out, None)
    return audio


def silence(num_samples: int, sample_width: int) -> bytes:
    return b"\x00" * max(0, int(num_samples)) * sample_width


def overlay_audio(dst: bytearray, src: bytes, start_sample: int, sample_width: int) -> None:
    start_byte = start_sample * sample_width
    end_byte = start_byte + len(src)
    mixed = audioop.add(bytes(dst[start_byte:end_byte]), src, sample_width)
    dst[start_byte:end_byte] = mixed


def patch_question_answer_audio(question_path: Path, answer_path: Path, answer_start_time: float) -> tuple[bytes, int, int]:
    question, question_sr, sample_width = load_wav_mono(question_path)
    answer, answer_sr, answer_sample_width = load_wav_mono(answer_path)
    answer = convert_wav_audio(answer, answer_sr, answer_sample_width, question_sr, sample_width)

    question_samples = len(question) // sample_width
    answer_samples = len(answer) // sample_width
    answer_start = question_samples + int(round(float(answer_start_time) * question_sr))

    if answer_start < 0:
        question = silence(-answer_start, sample_width) + question
        question_samples = len(question) // sample_width
        answer_start = 0

    total_samples = max(question_samples, answer_start + answer_samples)
    patched = bytearray(silence(total_samples, sample_width))
    overlay_audio(patched, question, 0, sample_width)
    overlay_audio(patched, answer, answer_start, sample_width)
    return bytes(patched), question_sr, sample_width


def write_patched_audio(row: dict[str, str]) -> Path:
    question_path = Path(safe_str(row.get("audio_path")))
    answer_path = Path(safe_str(row.get("answer_audio_path")))
    answer_start_time = float(safe_str(row.get("answer_start_time")) or 0.0)
    patched_audio, sr, sample_width = patch_question_answer_audio(question_path, answer_path, answer_start_time)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with wave.open(str(tmp_path), "wb") as outfile:
        outfile.setnchannels(1)
        outfile.setsampwidth(sample_width)
        outfile.setframerate(sr)
        outfile.writeframes(patched_audio)
    return tmp_path


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


METRIC_FIELDS = [
    "stance_choice",
    "stance_probability",
    "stance_score",
    "stance_evidence",
    "stance_raw_response",
    "stance_error",
]


def stance_row_key(row: dict[str, Any]) -> str:
    row_idx = safe_str(row.get("row_idx"))
    if row_idx:
        return row_idx
    return safe_str(row.get("audio_path"))


def successful_score(row: dict[str, Any]) -> bool:
    if safe_str(row.get("stance_error")):
        return False
    return all(safe_str(row.get(field)) for field in ("stance_choice", "stance_probability", "stance_score"))


def load_previous_scores(output_path: Path) -> dict[str, dict[str, str]]:
    if not output_path.exists():
        return {}
    return {
        stance_row_key(row): row
        for row in read_csv_rows(output_path)
        if stance_row_key(row)
    }


def merge_previous_score(row: dict[str, str], previous: dict[str, str]) -> dict[str, Any]:
    out = dict(row)
    for field in METRIC_FIELDS:
        out[field] = previous.get(field, "")
    return out


def failed_score(error: str, raw: str = "") -> dict[str, Any]:
    return {
        "stance_choice": "",
        "stance_probability": "",
        "stance_score": "",
        "stance_evidence": "[]",
        "stance_raw_response": raw,
        "stance_error": safe_str(error)[:2000],
    }


def score_row(row: dict[str, str], client: OpenAIChatClient, temperature: float, max_tokens: int) -> dict[str, Any]:
    question_path = Path(safe_str(row.get("audio_path")))
    answer_path = Path(safe_str(row.get("answer_audio_path")))
    if not question_path.exists():
        return failed_score(f"missing question audio: {question_path}")
    if not answer_path.exists():
        return failed_score(f"missing answer audio: {answer_path}")

    patched_path = None
    try:
        patched_path = write_patched_audio(row)
        raw = client.complete(audio_path=patched_path, prompt=build_prompt(row), temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        return failed_score(f"request_failed: {exc}")
    finally:
        if patched_path is not None:
            try:
                patched_path.unlink()
            except FileNotFoundError:
                pass
    obj = extract_json_object(raw)
    if obj is None:
        return failed_score("invalid_json", raw=raw)

    try:
        choice = safe_str(obj.get("choice")).upper()
        probability = float(obj.get("probability"))
        score = map_choice_probability_to_score(choice, probability)
        evidence = coerce_evidence(obj.get("evidence"))[:4]
    except Exception as exc:
        return failed_score(f"invalid_response_fields: {exc}", raw=raw)

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
    parser.add_argument("--force-recompute", action="store_true", help="Recompute all valid rows instead of only failed/missing rows.")
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
        write_csv_rows(output_path, [], original_fields + METRIC_FIELDS)
        print(f"Wrote STANCE metrics to {output_path}")
        return

    output_name = args.metrics_name or f"stance_metrics_Q{safe_str(valid_rows[0].get('question_index'))}.csv"
    output_path = args.outputs_dir / output_name
    previous_scores = {} if args.force_recompute else load_previous_scores(output_path)

    rows_to_score: list[dict[str, str]] = []
    reused_count = 0
    for row in valid_rows:
        previous = previous_scores.get(stance_row_key(row), {})
        if previous and successful_score(previous):
            reused_count += 1
        else:
            rows_to_score.append(row)

    if reused_count:
        print(f"Reusing {reused_count}/{len(valid_rows)} previously successful STANCE rows.", flush=True)
    if rows_to_score:
        print(f"Scoring {len(rows_to_score)}/{len(valid_rows)} STANCE rows.", flush=True)
    else:
        print("No STANCE rows need scoring.", flush=True)

    newly_scored: dict[str, dict[str, Any]] = {}
    if rows_to_score:
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

        for row in tqdm(rows_to_score, total=len(rows_to_score), desc="scoring STANCE rows"):
            result = score_row(row, client, args.temperature, args.max_tokens)
            out = dict(row)
            out.update(result)
            newly_scored[stance_row_key(row)] = out

    scored_rows: list[dict[str, Any]] = []
    for row in valid_rows:
        key = stance_row_key(row)
        if key in newly_scored:
            scored_rows.append(newly_scored[key])
        elif key in previous_scores and successful_score(previous_scores[key]):
            scored_rows.append(merge_previous_score(row, previous_scores[key]))
        else:
            scored_rows.append(dict(row))

    original_fields = list(valid_rows[0].keys())
    write_csv_rows(output_path, scored_rows, original_fields + METRIC_FIELDS)
    print(f"Wrote STANCE metrics to {output_path}")


if __name__ == "__main__":
    main()
