"""用 google.genai 把视频交给 Gemini。凭证只从环境变量读，跟 synth.vertex_speak 同一套，不打印密钥。

有 GEMINI_API_KEY 时走开发者 API（Files API 上传长视频）。否则有 GOOGLE_CLOUD_PROJECT 时走 Vertex，
和 TTS 的 vertex_speak 一样。短片直接内联；超过 15MB 走 Files API。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .review import (
    INLINE_MAX_BYTES,
    Chunk,
    ReviewError,
    Turn,
    estimate_tokens,
    redact,
)

_RESOLUTION = {"low": "media_resolution_low", "high": "media_resolution_high"}
_PROCESSING = {"static": "STATIC", "agentic": "AGENTIC"}


def client_mode() -> str:
    """gemini：开发者 API。vertex：和 TTS 一样的 Vertex 客户端。"""
    forced = os.environ.get("ANIMIND_REVIEW_BACKEND", "").strip().lower()
    if forced in {"gemini", "vertex"}:
        return forced
    if forced:
        raise ReviewError("ANIMIND_REVIEW_BACKEND 只能是 gemini 或 vertex")
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return "vertex"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    raise ReviewError(
        "缺少审片用的 Gemini 凭证。在 --env 里设置 GEMINI_API_KEY（开发者 API，长视频走 Files API），"
        "或设置 GOOGLE_CLOUD_PROJECT 和 GOOGLE_API_KEY（和 Vertex TTS 同一套）。密钥不会打印。")


def make_client():
    from google import genai

    mode = client_mode()
    if mode == "vertex":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ReviewError("Vertex 审片需要 GOOGLE_API_KEY，和 synth 的 Vertex TTS 一样。")
        return genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
            api_key=os.environ["GOOGLE_API_KEY"],
        )
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return genai.Client(api_key=key, http_options={"api_version": "v1beta"})


def is_transient(exc: BaseException) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in {408, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(token in text for token in ("429", "500", "502", "503", "504", "timeout", "temporarily", "resource exhausted"))


def call_with_retries(fn, *, attempts: int, sleep):
    """fn() 成功时返回值；429/5xx 按指数退避重试。非瞬时错误立刻抛出。"""
    if attempts < 1:
        raise ReviewError("重试次数至少为 1")
    delay = 1.0
    for index in range(attempts):
        try:
            return fn()
        except ReviewError:
            raise
        except Exception as exc:
            if not is_transient(exc) or index + 1 == attempts:
                raise
            sleep(delay)
            delay = min(delay * 2, 20.0)
    raise ReviewError("重试失败")


def tokens_from_usage(usage) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    tool = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
    output = int(getattr(usage, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
    return prompt + tool, output + thoughts


def _state_name(uploaded) -> str:
    state = getattr(uploaded, "state", None)
    if state is None:
        return "ACTIVE"
    return getattr(state, "name", None) or str(state)


def wait_until_active(client, uploaded, *, sleep, attempts: int = 60):
    name = getattr(uploaded, "name", None)
    for _ in range(attempts):
        state = _state_name(uploaded)
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise ReviewError("Gemini 处理视频文件失败")
        if not name:
            return uploaded
        sleep(2)
        uploaded = client.files.get(name=name)
    raise ReviewError("视频文件一直在处理，超时了")


def _inline_part(types, data: bytes, mime: str, resolution: dict, processing: str):
    blob = types.Blob(data=data, mime_type=mime)
    try:
        return types.Part(inline_data=blob, media_resolution=resolution, media_processing=processing)
    except TypeError:
        return types.Part(inline_data=blob)


def _file_part(types, uri: str, mime: str, resolution: dict, processing: str):
    """Files API 返回的地址。from_uri 不收 media_processing，所以直接组 Part。"""
    try:
        return types.Part(
            file_data=types.FileData(file_uri=uri, mime_type=mime),
            media_resolution=resolution,
            media_processing=processing,
        )
    except TypeError:
        return types.Part.from_uri(file_uri=uri, mime_type=mime, media_resolution=resolution)


def generate_text(client, model: str, contents, config) -> tuple[str, object]:
    """优先用流式，避免长视频把同步请求拖过认证窗口。返回 (文本, usage)。"""
    chunks: list[str] = []
    usage = None
    for piece in client.models.generate_content_stream(model=model, contents=contents, config=config):
        text = getattr(piece, "text", None)
        if text:
            chunks.append(text)
        meta = getattr(piece, "usage_metadata", None)
        if meta is not None:
            usage = meta
    return "".join(chunks), usage


class GeminiReviewer:
    def __init__(self, model: str, *, resolution: str = "high", processing: str = "static", attempts: int = 3,
                 sleep=None, client=None):
        if resolution not in _RESOLUTION:
            raise ReviewError("media resolution 只能是 low 或 high")
        if processing not in _PROCESSING:
            raise ReviewError("media processing 只能是 static 或 agentic")
        self.model = model
        self.resolution = resolution
        self.processing = processing
        self.attempts = attempts
        self.sleep = sleep or time.sleep
        self.client = client

    def __call__(self, *, video: Path, system: str, user: str, chunk: Chunk) -> Turn:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ReviewError("审片需要 google-genai：uv sync --extra review") from exc
        client = None
        uploaded_name = None
        calls: list[tuple[int, int]] = []
        source = "usage"
        try:
            client = self.client or make_client()
            part, uploaded_name = self._video_part(client, types, video)
            config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.2,
            )

            def once():
                nonlocal source
                try:
                    text, usage = generate_text(client, self.model, [part, user], config)
                except Exception:
                    calls.append((0, 0))
                    raise
                input_tokens, output_tokens = tokens_from_usage(usage)
                if input_tokens == 0 and output_tokens == 0:
                    input_tokens, output_tokens = estimate_tokens(system + user, chunk.length, self.resolution)
                    source = "estimate"
                calls.append((input_tokens, output_tokens))
                return text

            try:
                text = call_with_retries(once, attempts=self.attempts, sleep=self.sleep)
            except ReviewError:
                raise
            except Exception as exc:
                raise ReviewError(redact(f"Gemini 审片失败：{exc}"), calls=list(calls), token_source=source) from exc
            return Turn(text=text, calls=calls, token_source=source)
        finally:
            if client is not None:
                self._delete(client, uploaded_name)

    def _delete(self, client, name: str | None) -> None:
        if not name:
            return
        try:
            client.files.delete(name=name)
        except Exception:
            return

    def _video_part(self, client, types, video: Path):
        resolution = {"level": _RESOLUTION[self.resolution]}
        processing = _PROCESSING[self.processing]
        mime = "video/mp4"
        if video.stat().st_size <= INLINE_MAX_BYTES:
            data = video.read_bytes()
            return _inline_part(types, data, mime, resolution, processing), None
        try:
            uploaded = client.files.upload(file=str(video))
        except Exception as exc:
            raise ReviewError(redact(
                "这段超过 15MB，Files API 上传失败。设置 GEMINI_API_KEY 走开发者 API，或减小 --max-chunk-seconds。"
                f"（{exc}）")) from exc
        uploaded = wait_until_active(client, uploaded, sleep=self.sleep)
        name = getattr(uploaded, "name", None)
        uri = getattr(uploaded, "uri", None)
        if not uri:
            raise ReviewError("Files API 没有返回视频地址。可以改用 GEMINI_API_KEY，或减小 --max-chunk-seconds 让每段小于 15MB。")
        mime = getattr(uploaded, "mime_type", None) or mime
        return _file_part(types, uri, mime, resolution, processing), name
