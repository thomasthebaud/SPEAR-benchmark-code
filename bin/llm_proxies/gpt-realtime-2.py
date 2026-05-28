# llm_proxies/gpt-realtime-2.py

import base64
import io
import json
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import websocket
except ImportError as exc:
    raise ImportError(
        "The gpt-realtime-2 proxy requires websocket-client. "
        "Install it with `pip install websocket-client`."
    ) from exc


OPENAI_REALTIME_URL_TEMPLATE = "wss://api.openai.com/v1/realtime?model={model}"

TARGET_RATE = 24000
CHUNK_MS = 40
BYTES_PER_SAMPLE = 2
CHANNELS = 1

# Extra silence after the input file so server-side VAD can close the final turn.
TAIL_SILENCE_MS = 2500

# How long to wait after the input file is fully streamed.
POST_STREAM_WAIT_S = 30.0

# How long to give VAD to naturally finalize after EOF before forcing commit.
VAD_FINALIZE_GRACE_S = 3.0

DEFAULT_INSTRUCTIONS = (
    "You are participating in a natural spoken conversation.\n"
    "The user audio is being streamed to you in real time from a file.\n"
    "Answer when it feels natural, including before the entire audio file has finished if appropriate.\n"
    "Keep responses conversational and concise.\n"
    "If the user continues speaking or interrupts, stop focusing on the old response and respond to the latest user speech."
)


def _resample_mono_audio(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int = TARGET_RATE,
) -> np.ndarray:
    """Convert input audio to mono float32 at target_rate."""
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    samples = samples.astype(np.float32, copy=False)

    if samples.size == 0 or source_rate == target_rate:
        return samples

    output_length = max(1, round(samples.shape[0] * target_rate / source_rate))
    target_positions = np.linspace(0, samples.shape[0] - 1, num=output_length)

    return np.interp(
        target_positions,
        np.arange(samples.shape[0]),
        samples,
    ).astype(np.float32)


def _read_audio_as_pcm16(audio_path: Path) -> bytes:
    """Load an audio file and return mono PCM16 little-endian bytes at 24 kHz."""
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)

    mono = _resample_mono_audio(audio, sample_rate, TARGET_RATE)
    mono = np.clip(mono, -1.0, 1.0)

    pcm16 = (mono * 32767.0).astype("<i2")

    if TAIL_SILENCE_MS > 0:
        silence_frames = int(TARGET_RATE * TAIL_SILENCE_MS / 1000.0)
        tail = np.zeros(silence_frames, dtype="<i2")
        pcm16 = np.concatenate([pcm16, tail])

    return pcm16.tobytes()


def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = TARGET_RATE) -> bytes:
    """Wrap raw PCM16 bytes into a WAV container."""
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return buffer.getvalue()


def _json_send(ws: websocket.WebSocket, payload: Dict) -> None:
    ws.send(json.dumps(payload))


def _parse_server_messages(raw) -> Iterable[Dict]:
    """Parse one or more JSON server events from a websocket payload."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    raw = raw.strip()
    if not raw:
        return

    try:
        yield json.loads(raw)
        return
    except json.JSONDecodeError:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _connect(
    api_key: str,
    model_name: str,
    org: Optional[str] = None,
) -> websocket.WebSocket:
    url = OPENAI_REALTIME_URL_TEMPLATE.format(model=model_name or "gpt-realtime-2")

    headers = [f"Authorization: Bearer {api_key}"]

    if org:
        headers.append(f"OpenAI-Organization: {org}")

    return websocket.create_connection(url, header=headers)


def _configure_dynamic_session(
    ws: websocket.WebSocket,
    session_instructions: str,
) -> None:
    """
    Configure a dynamic realtime session.

    We use server_vad instead of semantic_vad because benchmark files often need
    explicit silence-based finalization. With create_response=True, the server can
    answer automatically when a turn is detected.
    """
    _json_send(
        ws,
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": session_instructions,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": TARGET_RATE,
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": TARGET_RATE,
                        },
                        "voice": "alloy",
                    },
                },
            },
        },
    )


def _receiver_loop(
    ws: websocket.WebSocket,
    audio_fragments: list,
    text_fragments: list,
    event_log: list,
    error_queue: queue.Queue,
    stop_event: threading.Event,
    response_done_event: threading.Event,
    active_responses: set,
    completed_responses: list,
    timing_state: dict,
) -> None:
    """
    Receive server events while audio is still being streamed.

    Important: do not treat the first response.done as final. In dynamic VAD mode,
    a response may be created, interrupted by later user audio, and then followed
    by another response.
    """
    ws.settimeout(1.0)

    while not stop_event.is_set():
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as exc:
            if not stop_event.is_set():
                error_queue.put(exc)
            return

        for event in _parse_server_messages(raw):
            event_type = event.get("type")
            event_log.append(event_type)

            if event_type == "response.created":
                response = event.get("response", {})
                response_id = response.get("id")
                if response_id:
                    active_responses.add(response_id)

            elif event_type == "response.output_audio.delta":
                delta = event.get("delta")
                if delta:
                    if timing_state["answer_start_s"] is None:
                        input_start_s = timing_state.get("input_stream_start_s")
                        if input_start_s is not None:
                            timing_state["answer_start_s"] = max(0.0, time.monotonic() - input_start_s)
                    audio_fragments.append(delta)

            elif event_type == "response.output_audio_transcript.delta":
                delta = event.get("delta")
                if delta:
                    text_fragments.append(delta)

            elif event_type == "response.text.delta":
                delta = event.get("delta") or event.get("text")
                if delta:
                    text_fragments.append(delta)

            elif event_type == "response.done":
                response = event.get("response", {})
                response_id = response.get("id")
                status = response.get("status")

                completed_responses.append(
                    {
                        "id": response_id,
                        "status": status,
                    }
                )

                if response_id and response_id in active_responses:
                    active_responses.remove(response_id)

                # Only mark done if no response is currently active.
                # Do not necessarily exit immediately; the main loop will wait
                # for audio or a quiet period.
                if not active_responses:
                    response_done_event.set()

            elif event_type == "error":
                error_queue.put(RuntimeError(json.dumps(event, indent=2)))
                stop_event.set()
                return


def _wait_for_session_updated(
    event_log: list,
    error_queue: queue.Queue,
    timeout_s: float = 10.0,
) -> None:
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if not error_queue.empty():
            raise error_queue.get()

        if "session.updated" in event_log:
            return

        time.sleep(0.05)

    raise TimeoutError("Timed out waiting for session.updated.")


def _force_final_response_if_needed(
    ws: websocket.WebSocket,
    event_log: list,
) -> None:
    """
    Fallback for file simulations where VAD starts but never stops.

    If the server detected speech_started but did not emit speech_stopped,
    committed, or any response events, force a final commit + response.create
    after EOF.
    """
    saw_started = "input_audio_buffer.speech_started" in event_log
    saw_stopped = "input_audio_buffer.speech_stopped" in event_log
    saw_committed = "input_audio_buffer.committed" in event_log
    saw_response = any(
        isinstance(evt, str) and evt.startswith("response.")
        for evt in event_log
    )

    if saw_response:
        return

    if saw_started and not saw_stopped and not saw_committed:
        _json_send(ws, {"type": "input_audio_buffer.commit"})
        _json_send(
            ws,
            {
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                },
            },
        )


def get_reply_with_audio(
    audio_path: Path,
    instruction: str,
    model_name: str,
    org: str,
    api_key: str,
    temp: float = 0.7,
) -> Tuple[Optional[bytes], str, Optional[str], bool, Optional[float]]:
    """
    SPEARBench-compatible realtime proxy.

    Args:
        audio_path: Path to input audio.
        instruction: Optional task/system prompt from the runner.
        model_name: Model name, usually "gpt-realtime-2".
        org: Optional OpenAI organization ID.
        api_key: OpenAI API key.
        temp: Kept for compatibility with runner; not sent to Realtime.

    Returns:
        audio_answer_bytes:
            WAV bytes containing all captured assistant audio, or None on failure.
        transcript_answer:
            Concatenated assistant transcript text if available.
        finish_reason:
            "completed", "timeout", "error: ...", or diagnostic reason.
        success:
            True iff usable assistant audio was captured.
    """
    del temp  # Realtime session rejects session.temperature.

    if not api_key:
        raise ValueError("api_key is required for gpt-realtime-2 proxy.")

    audio_path = Path(audio_path)

    if not audio_path.exists():
        return None, "", f"Audio file does not exist: {audio_path}", False, None

    try:
        pcm_bytes = _read_audio_as_pcm16(audio_path)
    except Exception as exc:
        return None, "", f"failed_to_read_audio: {exc}", False, None

    if not pcm_bytes:
        return None, "", "empty_audio", False, None

    session_instructions = DEFAULT_INSTRUCTIONS

    if instruction and instruction != "None":
        session_instructions = f"{DEFAULT_INSTRUCTIONS}\n\n{instruction}"

    ws = None
    receiver = None

    stop_event = threading.Event()
    response_done_event = threading.Event()
    error_queue: queue.Queue = queue.Queue()

    audio_fragments = []
    text_fragments = []
    event_log = []
    active_responses = set()
    completed_responses = []
    timing_state = {
        "input_stream_start_s": None,
        "answer_start_s": None,
    }

    try:
        ws = _connect(api_key=api_key, model_name=model_name, org=org)

        receiver = threading.Thread(
            target=_receiver_loop,
            args=(
                ws,
                audio_fragments,
                text_fragments,
                event_log,
                error_queue,
                stop_event,
                response_done_event,
                active_responses,
                completed_responses,
                timing_state,
            ),  
            daemon=True,
        )
        receiver.start()

        _configure_dynamic_session(ws, session_instructions)
        _wait_for_session_updated(event_log, error_queue)

        chunk_size = int(TARGET_RATE * BYTES_PER_SAMPLE * (CHUNK_MS / 1000.0))

        timing_state["input_stream_start_s"] = time.monotonic()

        for offset in range(0, len(pcm_bytes), chunk_size):
            if not error_queue.empty():
                raise error_queue.get()

            chunk = pcm_bytes[offset : offset + chunk_size]
            _json_send(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
            )

            # Real-time pacing; this is what makes the file behave like mic input.
            time.sleep(CHUNK_MS / 1000.0)

        # Give VAD a chance to naturally emit speech_stopped/committed/response.
        vad_deadline = time.time() + VAD_FINALIZE_GRACE_S

        while time.time() < vad_deadline:
            if not error_queue.empty():
                raise error_queue.get()

            if (
                "input_audio_buffer.speech_stopped" in event_log
                or "input_audio_buffer.committed" in event_log
                or response_done_event.is_set()
                or any(
                    isinstance(evt, str) and evt.startswith("response.")
                    for evt in event_log
                )
            ):
                break

            time.sleep(0.05)

        # If VAD got stuck after speech_started, force final batch-style response.
        _force_final_response_if_needed(ws, event_log)

        deadline = time.time() + POST_STREAM_WAIT_S
        last_audio_count = len(audio_fragments)
        last_text_count = len(text_fragments)
        last_activity_time = time.time()

        while time.time() < deadline:
            if not error_queue.empty():
                raise error_queue.get()

            current_audio_count = len(audio_fragments)
            current_text_count = len(text_fragments)

            if current_audio_count != last_audio_count or current_text_count != last_text_count:
                last_audio_count = current_audio_count
                last_text_count = current_text_count
                last_activity_time = time.time()

            # Exit only after at least some audio arrived and the response stream has
            # gone quiet. This avoids stopping on an interrupted/cancelled response.done.
            if audio_fragments and response_done_event.is_set():
                if time.time() - last_activity_time > 1.0:
                    break

            # If no audio yet but transcript is arriving, keep waiting.
            time.sleep(0.05)

        transcript = "".join(text_fragments).strip()

        if not audio_fragments:
            debug_tail = ", ".join(str(evt) for evt in event_log[-30:])
            return (
                None,
                transcript,
                f"no_audio_fragments; recent_events=[{debug_tail}]",
                False,
                None
            )

        try:
            raw_audio = b"".join(
                base64.b64decode(fragment)
                for fragment in audio_fragments
            )
        except Exception as exc:
            return None, transcript, f"failed_to_decode_audio: {exc}", False, None

        if not raw_audio:
            debug_tail = ", ".join(str(evt) for evt in event_log[-30:])
            return (
                None,
                transcript,
                f"empty_decoded_audio; recent_events=[{debug_tail}]",
                False,
                None,
            )

        wav_bytes = _pcm16_to_wav_bytes(raw_audio, TARGET_RATE)
        finish_reason = "completed" if response_done_event.is_set() else "timeout"

        return wav_bytes, transcript, finish_reason, True, timing_state["answer_start_s"]

    except Exception as exc:
        transcript = "".join(text_fragments).strip()
        return None, transcript, f"error: {exc}", False, None

    finally:
        stop_event.set()

        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

        if receiver is not None:
            try:
                receiver.join(timeout=2.0)
            except Exception:
                pass