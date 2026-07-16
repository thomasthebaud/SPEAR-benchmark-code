#!/usr/bin/env python3
"""Standalone full-duplex Nova Sonic audio QA debug using openai_keys.sh.

This intentionally inlines the working NovaSonicJudge request path from
run_turnover_nova_QA.py, without importing that module.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
import socket
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

NOVA_ALLOWED_SAMPLE_RATES = (8000, 16000)
NOVA_ALLOWED_ENDPOINTING_SENSITIVITY = ("LOW", "MEDIUM", "HIGH")
NOVA_ALLOWED_VOICE_IDS = (
    "ambre",
    "amy",
    "arjun",
    "beatrice",
    "carlos",
    "carolina",
    "florian",
    "kiara",
    "lennart",
    "leo",
    "lorenzo",
    "lupe",
    "matthew",
    "olivia",
    "tiffany",
    "tina",
)


def _normalize_nova_voice_id(value: Any, *, arg_name: str = "voice_id") -> str:
    """Normalize/validate Nova voice IDs against the documented allow-list."""
    v = str(value).strip().lower()
    if not v:
        raise ValueError(f"{arg_name} must not be empty.")
    if v not in NOVA_ALLOWED_VOICE_IDS:
        allowed = ", ".join(NOVA_ALLOWED_VOICE_IDS)
        raise ValueError(f"Invalid {arg_name}={v!r}. Allowed values: {allowed}")
    return v



class NovaSonicJudge:
    """
    Best-effort Amazon Nova 2 Sonic "model swap" wrapper.

    - Uses Amazon Bedrock Runtime bidirectional streaming API.
    - Sends (system text, user text, one-or-two audio segments) as one prompt.
    - Collects assistant TEXT output (SPECULATIVE first, then FINAL fallback).
    - Returns a raw string (expected to be JSON) for downstream parsing.

    Notes:
      * Nova Sonic is speech-to-speech; audio output may still be produced by the service.
        We ignore audio output events and only parse assistant textOutput.
      * This wrapper is intentionally minimal and avoids changing the downstream scoring logic.
    """

    def __init__(
        self,
        model_id: str,
        region: str,
        endpoint_uri: Optional[str] = None,
        voice_id: str = "matthew",
        input_sample_rate_hz: int = 16000,
        output_sample_rate_hz: int = 16000,
        audio_chunk_samples: int = 1024,
        send_sleep_s: float = 0.0,
        post_audio_wait_s: float = 2.0,
        timeout_s: float = 120.0,
        endpointing_sensitivity: str = "HIGH",
    ) -> None:
        self.model_id = str(model_id)
        self.region = str(region)
        self.endpoint_uri = str(endpoint_uri) if endpoint_uri else None
        self.voice_id = _normalize_nova_voice_id(voice_id, arg_name="voice_id")
        self.input_sample_rate_hz = int(input_sample_rate_hz)
        self.output_sample_rate_hz = int(output_sample_rate_hz)
        if self.input_sample_rate_hz not in NOVA_ALLOWED_SAMPLE_RATES:
            raise ValueError(
                f"Invalid input_sample_rate_hz={self.input_sample_rate_hz}. "
                f"Allowed: {NOVA_ALLOWED_SAMPLE_RATES}"
            )
        if self.output_sample_rate_hz not in NOVA_ALLOWED_SAMPLE_RATES:
            raise ValueError(
                f"Invalid output_sample_rate_hz={self.output_sample_rate_hz}. "
                f"Allowed: {NOVA_ALLOWED_SAMPLE_RATES}"
            )
        self.audio_chunk_samples = int(audio_chunk_samples)
        self.send_sleep_s = float(send_sleep_s)
        self.post_audio_wait_s = float(post_audio_wait_s)
        self.timeout_s = float(timeout_s)
        self.endpointing_sensitivity = str(endpointing_sensitivity).strip().upper()
        if self.endpointing_sensitivity not in NOVA_ALLOWED_ENDPOINTING_SENSITIVITY:
            raise ValueError(
                f"Invalid endpointing_sensitivity={self.endpointing_sensitivity!r}. "
                f"Allowed: {NOVA_ALLOWED_ENDPOINTING_SENSITIVITY}"
            )

        # Lazily initialized (within the event loop).
        self._client = None

        # Keep a private event loop to avoid creating/destroying one per request.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # SDK symbols (lazy imports).
        self._sdk = {}

    def close(self) -> None:
        loop = self._loop
        try:
            if loop is not None and not loop.is_closed():
                asyncio.set_event_loop(loop)

                async def _cleanup_pending() -> None:
                    # Give Smithy/AWS CRT stream-close callbacks a turn, then cancel
                    # anything still attached to this private per-judge event loop.
                    await asyncio.sleep(0)
                    current = asyncio.current_task()
                    pending = [
                        task
                        for task in asyncio.all_tasks()
                        if task is not current and not task.done()
                    ]
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                with contextlib.suppress(BaseException):
                    loop.run_until_complete(_cleanup_pending())
                with contextlib.suppress(BaseException):
                    loop.run_until_complete(loop.shutdown_asyncgens())
                with contextlib.suppress(BaseException):
                    loop.run_until_complete(loop.shutdown_default_executor())
                loop.close()
        finally:
            self._loop = None
            self._client = None
            self._sdk = {}

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        # If we're already inside an event loop (e.g., Jupyter), we don't try to re-enter it.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "NovaSonicJudge.judge() was called from within an active asyncio event loop. "
                "Use an async entrypoint or refactor to call judge_async()."
            )

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self._loop

    def judge(
        self,
        system_text: str,
        user_text: str,
        audio_paths: List[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, bytes]:
        loop = self._get_loop()
        coro = self._judge_async(
            system_text=system_text,
            user_text=user_text,
            audio_paths=audio_paths,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        # Do not apply a blanket timeout to the whole coroutine: sending longer audio
        # can legitimately exceed timeout_s due stream backpressure. Timeouts are
        # enforced in stream open/frame receive and completion waiting paths.
        return loop.run_until_complete(coro)

    async def _ensure_client(self):
        if self._client is not None:
            return self._client

        # Lazy import so the script can be inspected without the SDK installed.
        try:
            # SDK (package name may vary slightly across previews; we try the documented one first).
            from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
        except Exception:
            BedrockRuntimeClient = None  # type: ignore

        try:
            from aws_sdk_bedrock_runtime.client import InvokeModelWithBidirectionalStreamOperationInput
        except Exception:
            InvokeModelWithBidirectionalStreamOperationInput = None  # type: ignore

        try:
            from aws_sdk_bedrock_runtime.models import (
                InvokeModelWithBidirectionalStreamInputChunk,
                BidirectionalInputPayloadPart,
            )
        except Exception:
            InvokeModelWithBidirectionalStreamInputChunk = None  # type: ignore
            BidirectionalInputPayloadPart = None  # type: ignore

        try:
            from aws_sdk_bedrock_runtime.config import Config
        except Exception:
            Config = None  # type: ignore

        # Auth/credentials imports have moved between preview versions; try multiple layouts.
        EnvironmentCredentialsResolver = None
        for mod_name in (
            "smithy_aws_core.identity.environment",
            "smithy_aws_core.identity",
            "smithy_aws_core.credentials_resolvers.environment",
            "smithy_aws_core.credentials",
        ):
            try:
                _m = __import__(mod_name, fromlist=["EnvironmentCredentialsResolver"])
                EnvironmentCredentialsResolver = getattr(_m, "EnvironmentCredentialsResolver", None)
                if EnvironmentCredentialsResolver is not None:
                    break
            except Exception:
                continue

        HTTPAuthSchemeResolver = None
        SigV4AuthScheme = None
        try:
            from aws_sdk_bedrock_runtime.config import HTTPAuthSchemeResolver, SigV4AuthScheme  # type: ignore
        except Exception:
            try:
                from smithy_http.aio.identity import HTTPAuthSchemeResolver  # type: ignore
                from smithy_aws_core.identity.auth import SigV4AuthScheme  # type: ignore
            except Exception:
                HTTPAuthSchemeResolver = None  # type: ignore
                SigV4AuthScheme = None  # type: ignore

        missing = []
        if BedrockRuntimeClient is None:
            missing.append("aws_sdk_bedrock_runtime.client.BedrockRuntimeClient")
        if InvokeModelWithBidirectionalStreamOperationInput is None:
            missing.append("InvokeModelWithBidirectionalStreamOperationInput")
        if InvokeModelWithBidirectionalStreamInputChunk is None or BidirectionalInputPayloadPart is None:
            missing.append("InvokeModelWithBidirectionalStreamInputChunk/BidirectionalInputPayloadPart")
        if Config is None:
            missing.append("aws_sdk_bedrock_runtime.config.Config")
        if EnvironmentCredentialsResolver is None:
            missing.append("EnvironmentCredentialsResolver")
        if HTTPAuthSchemeResolver is None or SigV4AuthScheme is None:
            missing.append("HTTPAuthSchemeResolver/SigV4AuthScheme")

        if missing:
            raise RuntimeError(
                "Missing Amazon Nova bidirectional streaming SDK dependencies. "
                "Install the Amazon Nova 2 Sonic Bedrock Runtime preview SDK (aws_sdk_bedrock_runtime) "
                f"and its smithy dependencies. Missing: {missing}"
            )

        endpoint_uri = self.endpoint_uri or f"https://bedrock-runtime.{self.region}.amazonaws.com"
        endpoint_host = ""
        try:
            endpoint_host = str(urlparse(endpoint_uri).hostname or "").strip()
        except Exception:
            endpoint_host = ""
        if endpoint_host:
            try:
                socket.getaddrinfo(endpoint_host, 443, type=socket.SOCK_STREAM)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot resolve Bedrock endpoint host '{endpoint_host}'. "
                    "Check DNS/network on this node or override --bedrock-endpoint-uri."
                ) from e

        # Config constructor signature differs between preview versions; attempt multiple layouts.
        def _mk_sigv4():
            # Newer previews require SigV4AuthScheme(service="bedrock")
            try:
                return SigV4AuthScheme(service="bedrock")
            except TypeError:
                return SigV4AuthScheme()

        config = None
        config_errors: List[str] = []

        resolver_inst = None
        if EnvironmentCredentialsResolver is not None:
            try:
                resolver_inst = EnvironmentCredentialsResolver()
            except Exception as e:
                config_errors.append(f"EnvironmentCredentialsResolver() failed: {type(e).__name__}: {e}")
                resolver_inst = None

        # Prefer the Nova 2 signature from the AWS guide: auth_scheme_resolver/auth_schemes
        config_builders = []

        if resolver_inst is not None and HTTPAuthSchemeResolver is not None and SigV4AuthScheme is not None:
            config_builders.append(
                lambda: Config(
                    endpoint_uri=endpoint_uri,
                    region=self.region,
                    aws_credentials_identity_resolver=resolver_inst,
                    auth_scheme_resolver=HTTPAuthSchemeResolver(),
                    auth_schemes={"aws.auth#sigv4": _mk_sigv4()},
                )
            )
            # Older preview naming (http_auth_*)
            config_builders.append(
                lambda: Config(
                    endpoint_uri=endpoint_uri,
                    region=self.region,
                    aws_credentials_identity_resolver=resolver_inst,
                    http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
                    http_auth_schemes={"aws.auth#sigv4": _mk_sigv4()},
                )
            )

        # Region name variant fallback
        if resolver_inst is not None and HTTPAuthSchemeResolver is not None and SigV4AuthScheme is not None:
            config_builders.append(
                lambda: Config(
                    endpoint_uri=endpoint_uri,
                    region_name=self.region,
                    credentials_resolver=resolver_inst,
                    auth_scheme_resolver=HTTPAuthSchemeResolver(),
                    auth_schemes={"aws.auth#sigv4": _mk_sigv4()},
                )
            )
            config_builders.append(
                lambda: Config(
                    endpoint_uri=endpoint_uri,
                    region_name=self.region,
                    credentials_resolver=resolver_inst,
                    http_auth_scheme_resolver=HTTPAuthSchemeResolver(
                        schemes={"aws.auth#sigv4": _mk_sigv4()}
                    ),
                )
            )

        # Minimal fallback (may fail later if request signing is needed)
        config_builders.append(lambda: Config(endpoint_uri=endpoint_uri, region=self.region))
        config_builders.append(lambda: Config(endpoint_uri=endpoint_uri, region_name=self.region))

        for build_config in config_builders:
            try:
                config = build_config()
                break
            except Exception as e:
                config_errors.append(f"{type(e).__name__}: {e}")

        if config is None:
            raise RuntimeError(
                "Failed to construct Bedrock Runtime Config for installed SDK version. "
                f"Tried multiple signatures; last errors: {config_errors[-3:]}"
            )

        self._sdk = {
            "BedrockRuntimeClient": BedrockRuntimeClient,
            "InvokeModelWithBidirectionalStreamOperationInput": InvokeModelWithBidirectionalStreamOperationInput,
            "InvokeModelWithBidirectionalStreamInputChunk": InvokeModelWithBidirectionalStreamInputChunk,
            "BidirectionalInputPayloadPart": BidirectionalInputPayloadPart,
        }
        self._client = BedrockRuntimeClient(config=config)
        return self._client

    async def _send_event(self, stream, payload_obj: Dict[str, Any]) -> None:
        payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
        chunk = self._sdk["InvokeModelWithBidirectionalStreamInputChunk"](
            value=self._sdk["BidirectionalInputPayloadPart"](bytes_=payload)
        )
        send_timeout_s = max(30.0, float(self.timeout_s))
        try:
            await asyncio.wait_for(stream.input_stream.send(chunk), timeout=send_timeout_s)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Timed out sending event to Nova stream after {send_timeout_s:.1f}s."
            ) from e

    async def _stream_audio(
        self,
        stream,
        *,
        prompt_name: str,
        content_name: str,
        pcm16_bytes: bytes,
        end_content: bool = True,
    ) -> None:
        # audioInputConfiguration contentStart
        await self._send_event(
            stream,
            {
                "event": {
                    "contentStart": {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": int(self.input_sample_rate_hz),
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            },
        )

        # Stream PCM16 bytes as base64 payloads.
        bytes_per_sample = 2  # int16
        chunk_bytes = max(1, int(self.audio_chunk_samples) * bytes_per_sample)
        offset = 0
        while offset < len(pcm16_bytes):
            chunk = pcm16_bytes[offset : offset + chunk_bytes]
            offset += len(chunk)
            b64 = base64.b64encode(chunk).decode("utf-8")
            await self._send_event(
                stream,
                {
                    "event": {
                        "audioInput": {
                            "promptName": prompt_name,
                            "contentName": content_name,
                            "content": b64,
                        }
                    }
                },
            )
            if self.send_sleep_s > 0:
                await asyncio.sleep(self.send_sleep_s)

        if end_content:
            await self._send_event(
                stream,
                {"event": {"contentEnd": {"promptName": prompt_name, "contentName": content_name}}},
            )

    async def _send_audio_bytes(
        self,
        stream,
        *,
        prompt_name: str,
        content_name: str,
        pcm16_bytes: bytes,
    ) -> None:
        chunk_bytes = max(1, int(self.audio_chunk_samples) * 2)
        offset = 0
        while offset < len(pcm16_bytes):
            chunk = pcm16_bytes[offset : offset + chunk_bytes]
            offset += len(chunk)
            await self._send_event(
                stream,
                {
                    "event": {
                        "audioInput": {
                            "promptName": prompt_name,
                            "contentName": content_name,
                            "content": base64.b64encode(chunk).decode("utf-8"),
                        }
                    }
                },
            )
            if self.send_sleep_s > 0:
                await asyncio.sleep(self.send_sleep_s)

    async def _send_user_text(
        self,
        stream,
        *,
        prompt_name: str,
        text: str,
    ) -> None:
        user_content = f"user_{uuid.uuid4().hex}"
        await self._send_event(
            stream,
            {
                "event": {
                    "contentStart": {
                        "promptName": prompt_name,
                        "contentName": user_content,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "USER",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            },
        )
        await self._send_event(
            stream,
            {
                "event": {
                    "textInput": {
                        "promptName": prompt_name,
                        "contentName": user_content,
                        "content": text,
                    }
                }
            },
        )
        await self._send_event(
            stream,
            {"event": {"contentEnd": {"promptName": prompt_name, "contentName": user_content}}},
        )

    async def _judge_async(
        self,
        *,
        system_text: str,
        user_text: str,
        audio_paths: List[str],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, bytes]:
        """
        One Nova request:
          - system prompt (text)
          - user prompt (text)
          - 1 or 2 user audio content blocks (in-order)
          - collect assistant textOutput (SPECULATIVE preferred)
        """
        client = await self._ensure_client()

        # If sampling disabled, make the request as deterministic as Nova allows.
        if not do_sample:
            temperature = 0.0
            top_p = 1.0

        prompt_name = f"eval_{uuid.uuid4().hex}"

        op_input = self._sdk["InvokeModelWithBidirectionalStreamOperationInput"](model_id=self.model_id)
        stream = await client.invoke_model_with_bidirectional_stream(op_input)

        assistant_spec_text_parts: List[str] = []
        assistant_final_text_parts: List[str] = []
        assistant_audio_parts: List[bytes] = []
        event_key_counts: Dict[str, int] = {}

        # Receive loop: collect assistant text until completionEnd
        async def _recv_loop() -> None:
            content_role: Dict[str, str] = {}
            content_stage: Dict[str, str] = {}
            seen_texts: set[str] = set()

            async def _recv_frame() -> Any:
                """
                Receive one frame from output stream across preview-SDK variants.
                """
                # Most common shape in recent previews.
                out_stream = getattr(stream, "output_stream", None)
                if out_stream is not None and hasattr(out_stream, "receive"):
                    return await out_stream.receive()

                # Alternate shape used by some generated clients.
                if hasattr(stream, "await_output"):
                    out = await stream.await_output()
                    if isinstance(out, (tuple, list)):
                        for part in out:
                            if hasattr(part, "receive"):
                                return await part.receive()
                    if hasattr(out, "receive"):
                        return await out.receive()
                    out2 = getattr(out, "output_stream", None)
                    if out2 is not None and hasattr(out2, "receive"):
                        return await out2.receive()

                if hasattr(stream, "receive"):
                    return await stream.receive()

                raise RuntimeError(
                    f"Unsupported bidirectional stream interface: {type(stream).__name__}"
                )

            def _extract_payload_bytes(result_obj: Any) -> Optional[bytes]:
                """
                Pull raw payload bytes from various SDK frame/result wrappers.
                """
                candidates: List[Any] = [result_obj]
                v = getattr(result_obj, "value", None)
                if v is not None:
                    candidates.append(v)

                for c in candidates:
                    if c is None:
                        continue

                    if isinstance(c, (bytes, bytearray)):
                        return bytes(c)

                    if isinstance(c, str):
                        return c.encode("utf-8")

                    b = getattr(c, "bytes_", None)
                    if isinstance(b, (bytes, bytearray)):
                        return bytes(b)

                    b2 = getattr(c, "bytes", None)
                    if isinstance(b2, (bytes, bytearray)):
                        return bytes(b2)

                    if isinstance(c, dict):
                        for key in ("bytes_", "bytes", "payload", "content"):
                            dv = c.get(key, None)
                            if isinstance(dv, (bytes, bytearray)):
                                return bytes(dv)
                            if isinstance(dv, str):
                                return dv.encode("utf-8")

                return None

            def _extract_stream_error(result_obj: Any) -> Optional[str]:
                """
                Surface typed Bedrock output-exception frames that do not carry bytes payloads.
                """
                candidates: List[Any] = [result_obj]
                v = getattr(result_obj, "value", None)
                if v is not None:
                    candidates.append(v)

                for c in candidates:
                    if c is None:
                        continue

                    cls_name = type(c).__name__
                    msg = getattr(c, "message", None)
                    if isinstance(msg, str) and msg.strip():
                        pieces = [msg.strip()]
                        original_message = getattr(c, "original_message", None)
                        if isinstance(original_message, str) and original_message.strip():
                            pieces.append(f"original_message={original_message.strip()}")
                        original_status_code = getattr(c, "original_status_code", None)
                        if original_status_code is not None:
                            pieces.append(f"original_status_code={original_status_code}")
                        return f"{cls_name}: {'; '.join(pieces)}"

                    if isinstance(c, dict):
                        dmsg = c.get("message", None)
                        if isinstance(dmsg, str) and dmsg.strip():
                            return f"{cls_name}: {dmsg.strip()}"

                # Union wrappers usually carry the concrete member in the class name.
                union_name = type(result_obj).__name__
                if "Exception" in union_name:
                    inner = getattr(result_obj, "value", None)
                    inner_msg = getattr(inner, "message", None)
                    if isinstance(inner_msg, str) and inner_msg.strip():
                        return f"{union_name}: {inner_msg.strip()}"
                    return union_name

                tag = getattr(result_obj, "tag", None)
                if tag:
                    return f"{union_name}(tag={tag})"

                return None

            def _collect_text_candidates(obj: Any, out: List[str]) -> None:
                """Best-effort extraction for SDK/event schema drift."""
                if isinstance(obj, dict):
                    # Common text-carrier shapes
                    for k in ("content", "text", "outputText", "transcript", "value"):
                        v = obj.get(k, None)
                        if isinstance(v, str):
                            t = v.strip()
                            # Avoid obvious non-model text payloads.
                            if t and len(t) <= 8000 and ("\n" in t or "{" in t or len(t.split()) > 2):
                                out.append(t)
                    for v in obj.values():
                        _collect_text_candidates(v, out)
                elif isinstance(obj, list):
                    for v in obj:
                        _collect_text_candidates(v, out)

            while True:
                try:
                    result = await asyncio.wait_for(_recv_frame(), timeout=self.timeout_s)
                except asyncio.TimeoutError as e:
                    raise RuntimeError(
                        f"Timed out waiting for Nova output frame after {self.timeout_s:.1f}s."
                    ) from e

                stream_err = _extract_stream_error(result)
                if stream_err:
                    raise RuntimeError(stream_err)

                raw_payload = _extract_payload_bytes(result)
                if not raw_payload:
                    continue

                try:
                    data = json.loads(raw_payload.decode("utf-8"))
                except Exception:
                    continue

                # Some SDK versions wrap payload under {"event": ...}; others emit event fields top-level.
                event = data.get("event", None) if isinstance(data, dict) else None
                if isinstance(event, dict):
                    event_obj = event
                elif isinstance(data, dict):
                    event_obj = data
                else:
                    continue

                for k in event_obj.keys():
                    sk = str(k)
                    event_key_counts[sk] = event_key_counts.get(sk, 0) + 1

                if "contentStart" in event_obj:
                    cs = event_obj.get("contentStart", {})
                    if isinstance(cs, dict):
                        content_id = str(
                            cs.get("contentId") or cs.get("contentName") or cs.get("id") or ""
                        ).strip()
                        role = str(cs.get("role") or "").upper()
                        amf = cs.get("additionalModelFields")
                        stage = None
                        if isinstance(amf, str) and amf.strip():
                            try:
                                stage = json.loads(amf).get("generationStage")
                            except Exception:
                                stage = None
                        stage_s = str(stage).upper() if stage else ""
                        if content_id:
                            if role:
                                content_role[content_id] = role
                            if stage_s:
                                content_stage[content_id] = stage_s
                    continue

                if "textOutput" in event_obj:
                    to = event_obj.get("textOutput", {})
                    if not isinstance(to, dict):
                        continue
                    txt = str(to.get("content") or "")
                    if not txt:
                        continue

                    content_id = str(
                        to.get("contentId") or to.get("contentName") or to.get("id") or ""
                    ).strip()
                    role = str(to.get("role") or "").upper()
                    if not role and content_id:
                        role = content_role.get(content_id, "")
                    if role == "USER":
                        continue
                    if role != "ASSISTANT":
                        continue

                    stage = content_stage.get(content_id, "") if content_id else ""
                    if txt not in seen_texts:
                        seen_texts.add(txt)
                        if stage == "FINAL":
                            assistant_final_text_parts.append(txt)
                        else:
                            # SPECULATIVE is the typical "text response" stream; treat all non-FINAL as preferred.
                            assistant_spec_text_parts.append(txt)
                    continue

                if "audioOutput" in event_obj:
                    ao = event_obj.get("audioOutput", {})
                    if isinstance(ao, dict):
                        content = ao.get("content")
                        if isinstance(content, str) and content:
                            assistant_audio_parts.append(base64.b64decode(content))
                    continue

                if "completionEnd" in event_obj:
                    break

        # Full duplex: run receiver and sender as independent concurrent tasks.
        # This allows Nova audio/text output to arrive while input audio, silence,
        # and the text question are still being streamed.
        async def _send_loop() -> None:
            # Send events.
            payload_session_start = {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": int(max_new_tokens),
                            "topP": float(top_p),
                            "temperature": float(temperature),
                        },
                        "turnDetectionConfiguration": {
                            "endpointingSensitivity": str(self.endpointing_sensitivity).upper(),
                        },
                    }
                }
            }
            payload_prompt_start = {
                "event": {
                    "promptStart": {
                        "promptName": prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": int(self.output_sample_rate_hz),
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": self.voice_id,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            }
            await self._send_event(stream, payload_session_start)
            await self._send_event(stream, payload_prompt_start)

            # SYSTEM text block (must be first content block for Nova).
            sys_content = f"sys_{uuid.uuid4().hex}"
            await self._send_event(
                stream,
                {
                    "event": {
                        "contentStart": {
                            "promptName": prompt_name,
                            "contentName": sys_content,
                            "type": "TEXT",
                            "interactive": False,
                            "role": "SYSTEM",
                            "textInputConfiguration": {"mediaType": "text/plain"},
                        }
                    }
                },
            )
            await self._send_event(
                stream,
                {
                    "event": {
                        "textInput": {
                            "promptName": prompt_name,
                            "contentName": sys_content,
                            "content": str(system_text),
                        }
                    }
                },
            )
            await self._send_event(
                stream,
                {"event": {"contentEnd": {"promptName": prompt_name, "contentName": sys_content}}},
            )

            # Audio blocks in-order. The optional text question is sent after audio
            # data but before audio contentEnd, keeping cross-modal input in an
            # active audio stream as Nova 2 Sonic expects.
            user_text_s = str(user_text).strip()
            sent_text = False
            for ap_i, ap in enumerate(audio_paths):
                # Load as float32 mono at self.input_sample_rate_hz.
                x, sr = _load_prepared_audio(Path(ap), self.input_sample_rate_hz)
                if x.size == 0:
                    continue
                x = np.clip(x, -1.0, 1.0)
                pcm16 = (x * 32767.0).astype(np.int16).tobytes()

                audio_content = f"aud_{uuid.uuid4().hex}"
                is_last_audio = ap_i == (len(audio_paths) - 1)
                keep_open_for_text = bool(user_text_s) and is_last_audio
                await self._stream_audio(
                    stream,
                    prompt_name=prompt_name,
                    content_name=audio_content,
                    pcm16_bytes=pcm16,
                    end_content=not keep_open_for_text,
                )

                if keep_open_for_text:
                    silence_pre = np.zeros(int(self.input_sample_rate_hz * 0.5), dtype=np.int16).tobytes()
                    await self._send_audio_bytes(
                        stream,
                        prompt_name=prompt_name,
                        content_name=audio_content,
                        pcm16_bytes=silence_pre,
                    )
                    await self._send_user_text(stream, prompt_name=prompt_name, text=user_text_s)
                    sent_text = True
                    if self.post_audio_wait_s > 0:
                        silence_post = np.zeros(
                            int(self.input_sample_rate_hz * self.post_audio_wait_s),
                            dtype=np.int16,
                        ).tobytes()
                        await self._send_audio_bytes(
                            stream,
                            prompt_name=prompt_name,
                            content_name=audio_content,
                            pcm16_bytes=silence_post,
                        )
                    await self._send_event(
                        stream,
                        {"event": {"contentEnd": {"promptName": prompt_name, "contentName": audio_content}}},
                    )

            if user_text_s and not sent_text:
                await self._send_user_text(stream, prompt_name=prompt_name, text=user_text_s)

            # End prompt + session, then close input stream.
            await self._send_event(stream, {"event": {"promptEnd": {"promptName": prompt_name}}})
            await self._send_event(stream, {"event": {"sessionEnd": {}}})

        recv_task = asyncio.create_task(_recv_loop(), name="nova_recv_loop")
        send_task = asyncio.create_task(_send_loop(), name="nova_send_loop")

        try:
            done, _pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            if recv_task in done:
                # If the service emits an exception while we are still sending,
                # stop the sender and surface the real receive-side error.
                recv_task.result()
                if not send_task.done():
                    send_task.cancel()
                    with contextlib.suppress(BaseException):
                        await send_task
            if send_task in done:
                send_task.result()

            if not send_task.done():
                await send_task
        except Exception as send_exc:
            # Give receiver a short chance to surface concrete service/transport errors first.
            recv_exc: Optional[BaseException] = None
            if not recv_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(recv_task), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    recv_exc = e

            if recv_exc is None and recv_task.done():
                try:
                    recv_task.result()
                except Exception as e:
                    recv_exc = e

            if not send_task.done():
                send_task.cancel()
                with contextlib.suppress(BaseException):
                    await send_task
            if not recv_task.done():
                recv_task.cancel()
                with contextlib.suppress(BaseException):
                    await recv_task

            if recv_exc is not None:
                raise recv_exc from send_exc
            raise

        # Wait for receiver (bounded).
        recv_error: Optional[str] = None
        try:
            await asyncio.wait_for(recv_task, timeout=self.timeout_s)
        except asyncio.TimeoutError:
            recv_task.cancel()
            recv_error = f"timeout_after_{self.timeout_s:.1f}s"
        except Exception as e:
            recv_error = f"{type(e).__name__}: {e}"
        finally:
            with contextlib.suppress(Exception):
                await stream.input_stream.close()

        # Prefer speculative (text response) if present, else FINAL (transcribed spoken output).
        spec_text = "".join(assistant_spec_text_parts).strip()
        final_text = "".join(assistant_final_text_parts).strip()
        out = spec_text if spec_text else final_text
        audio_bytes = b"".join(assistant_audio_parts)
        if out or audio_bytes:
            return out, audio_bytes

        # Diagnostics when the stream produced events but no text payload.
        top_keys = sorted(event_key_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        key_summary = ",".join(f"{k}:{v}" for k, v in top_keys) if top_keys else "none"
        if recv_error and "AccessDeniedException" in recv_error:
            raise PermissionError(
                "Bedrock AccessDeniedException while invoking Nova bidirectional stream. "
                f"model_id={self.model_id} region={self.region}. "
                "Grant bedrock:InvokeModelWithBidirectionalStream (and often bedrock:InvokeModel), "
                "and ensure this account has model access enabled for the model in this region."
            )
        if recv_error:
            return f"[empty_nova_response_text events={key_summary} recv_error={recv_error}]", b"" 
        return f"[empty_nova_response_text events={key_summary}]", b""



def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    m = float(np.max(np.abs(x)))
    if m > 0:
        x = x / m
    return x.astype(np.float32)


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Lightweight resampler to avoid librosa/numba runtime issues in judge path."""
    x = x.astype(np.float32, copy=False)
    if x.size == 0 or int(sr_in) == int(sr_out):
        return x.astype(np.float32)
    n_out = int(round(len(x) * (float(sr_out) / float(sr_in))))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32)


def _load_prepared_audio(audio_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(audio_path), always_2d=True)
    if data.ndim != 2:
        raise RuntimeError(f"Expected 2-D audio array for {audio_path}, got shape {data.shape}.")
    mono = np.mean(data, axis=1).astype(np.float32)
    if int(sr) != int(target_sr):
        mono = _resample_linear(mono, int(sr), int(target_sr))
        sr = int(target_sr)
    mono = normalize_audio(mono)
    return mono, int(sr)




ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / "openai_keys.sh"
AUDIO_PATH = Path("data/seamless_2t_2s_questions/inputs/test/improvised/audios/V00_S2017_I00001160_P1273A_P2072A_0.wav").resolve()

SYSTEM_TEXT = (
    "You are a helpful audio QA assistant. Listen to the user's audio and answer "
    "the text question directly. Do not transcribe the audio unless asked."
)
USER_TEXT = (
    "You are participating in a natural spoken conversation. "
    "Answer when it feels natural, not only at the very end. "
    "Keep responses conversational and concise. "
    "If the user interrupts, stop and respond to the latest user speech."
)


def load_env(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


def write_wav(path: Path, pcm16_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)


def pcm16_to_wav_bytes(pcm16_bytes: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)
    return buffer.getvalue()


def get_reply_with_audio(
    audio_path: Path,
    instruction: str,
    model_name: str,
    org: str,
    api_key: str,
    temp: float = 0.0,
    verbose: bool = False,
) -> Tuple[Optional[bytes], str, Optional[str], bool, Optional[float]]:
    """Adapter used by bin/run_LLM_inference.py."""

    if verbose: print(f"Debug check of arguments: audio_path={audio_path} instruction={instruction} model_name={model_name}", flush=True)
    load_env(ENV_FILE)

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        return None, "", f"Audio file does not exist: {audio_path}", False, None
    if verbose: print(f"Loading the judge for model amazon.nova-2-sonic-v1:0", flush=True)
    judge = NovaSonicJudge(
        model_id="amazon.nova-2-sonic-v1:0",
        region="us-east-1",
        endpoint_uri=None,
        voice_id="matthew",
        input_sample_rate_hz=16000,
        output_sample_rate_hz=16000,
        audio_chunk_samples=512,
        send_sleep_s=0.032,
        post_audio_wait_s=4.0,
        timeout_s=60.0,
        endpointing_sensitivity="HIGH",
    )
    if verbose: print(f"Model loaded, inference time!", flush=True)
    try:
        raw_output, audio_bytes = judge.judge(
            system_text=SYSTEM_TEXT,
            user_text=instruction,
            audio_paths=[str(audio_path)],
            max_new_tokens=int(os.environ.get("NOVA_MAX_TOKENS", "128")),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
    finally:
        judge.close()
    if verbose: print("Inference successful.", flush=True)
    if not audio_bytes:
        if verbose: print(f"Warning: No audio output from Nova Sonic model {judge.model_id}. Transcript is: {raw_output}", flush=True)
        return None, raw_output, "no_audio_output", False, None

    wav_audio_bytes = pcm16_to_wav_bytes(audio_bytes, judge.output_sample_rate_hz)
    if verbose:
        print(
            f"Wrapped Nova LPCM output into WAV: raw_bytes={len(audio_bytes)} "
            f"wav_bytes={len(wav_audio_bytes)} sample_rate={judge.output_sample_rate_hz}",
            flush=True,
        )

    return wav_audio_bytes, raw_output, "completed", True, None
