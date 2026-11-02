from __future__ import annotations

import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import tiktoken

from services.account_service import account_service
from services.config import config
from services.image_storage_service import image_storage_service
from services.openai_backend_api import (
    ImageContentPolicyError,
    ImagePollTimeoutError,
    ImageStreamHardTimeoutError,
    OpenAIBackendAPI,
    TEXT_STREAM_TIMEOUT_SECS,
)
from utils.helper import (
    extract_image_from_message_content,
    is_codex_image_model,
    is_supported_image_model,
    split_image_model,
    supported_image_models,
    UpstreamHTTPError,
)
from utils.image_tokens import count_image_content_tokens
from utils.log import logger
from utils.ui_markup import strip_ui_markup


class ImageGenerationError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_type: str = "server_error",
        code: str | None = "upstream_error",
        param: str | None = None,
        account_email: str = "",
        conversation_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.account_email = account_email
        self.conversation_id = conversation_id

    def to_openai_error(self) -> dict[str, Any]:
        error_dict = {
            "error": {
                "message": public_image_error_message(str(self)),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }
        if self.account_email:
            error_dict["error"]["account_email"] = self.account_email
        return error_dict


def public_image_error_message(message: str) -> str:
    text = str(message or "").strip()
    lower = text.lower()
    if any(item in lower for item in ("backend-api/", "status=", "body=", "chatgpt.com", "upstreamhttperror")):
        return "The image generation request failed. Please try again later."
    return text or "The image generation request failed. Please try again later."


def is_token_invalid_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "token_invalidated" in text
        or "token_revoked" in text
        or "authentication token has been invalidated" in text
        or "invalidated oauth token" in text
    )


def is_tls_connection_error(message: str) -> bool:
    """检测 TLS/SSL 连接错误，这类错误通常可以通过重试解决。"""
    text = str(message or "").lower()
    return (
        "curl: (35)" in text
        or "tls connect error" in text
        or "openssl_internal" in text
        or "ssl: wrong_version_number" in text
        or "ssl: certificate_verify_failed" in text
        or "connection aborted" in text
        or "remote disconnected" in text
        or "connection reset by peer" in text
        or "curl: (92)" in text
        or ("http/2 stream" in text and "internal_error" in text)
    )


def is_connection_timeout_error(message: str) -> bool:
    """检测连接超时错误（如 curl 28），这类错误可通过同账号短等待重试解决。"""
    text = str(message or "").lower()
    return (
        "curl: (28)" in text
        or "operation timed out" in text
        or "connection timed out" in text
        or "read timed out" in text
        or "connect timeout" in text
    )


def image_stream_error_message(message: str) -> str:
    text = str(message or "")
    if is_token_invalid_error(text):
        return "image generation failed"
    if is_tls_connection_error(text):
        return "upstream image connection failed, please retry later"
    if is_connection_timeout_error(text):
        return "upstream connection timed out, please retry later"
    return text or "image generation failed"


def is_image_concurrency_error(error: BaseException) -> bool:
    """识别文件服务返回的账号级并发限流错误。"""
    if not isinstance(error, UpstreamHTTPError) or error.status_code not in {429, 503}:
        return False
    body = error.body
    try:
        body_text = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)
    except (TypeError, ValueError):
        body_text = repr(body)
    text = body_text.lower()
    return "concurrency_limit" in text or "too many concurrent" in text or "throttled" in text


def _image_remaining_timeout_secs(backend: object) -> float:
    deadline = getattr(backend, "image_request_deadline", None)
    if deadline is not None:
        try:
            return max(0.0, float(deadline) - time.monotonic())
        except (TypeError, ValueError):
            pass
    return float(config.image_poll_timeout_secs)


def _sleep_image_retry(backend: object, seconds: float) -> None:
    sleep_for = min(max(0.0, seconds), _image_remaining_timeout_secs(backend))
    if sleep_for > 0:
        time.sleep(sleep_for)


def _sleep_until_deadline(deadline: float, seconds: float) -> None:
    sleep_for = min(max(0.0, seconds), max(0.0, deadline - time.monotonic()))
    if sleep_for > 0:
        time.sleep(sleep_for)


REFERENCED_IMAGE_IDS_RE = re.compile(r'"referenced_image_ids"\s*:\s*\[([^\]]+)\]')
# 检测模型返回的部分工具调用 JSON（如 {"size":"1920x1088","n":1}）
# 这些 JSON 包含图片生成工具的参数，但没有实际生成图片
TOOL_PARAMS_JSON_RE = re.compile(
    r'\{\s*"size"\s*:\s*"\d+x\d+"\s*,\s*"n"\s*:\s*\d+\s*\}'
)


def is_model_text_reply_instead_of_image(message: str) -> bool:
    """检测模型是否返回了文本回复（包含工具调用 JSON）而非实际生成图片。

    当上游 ChatGPT 未能触发图片生成工具时，会返回一段描述性文本，
    其中可能包含 JSON 参数（如 prompt、referenced_image_ids、size/n 等）。
    这种情况应被视为「上游未生成图片」而非「内容策略违规」。

    检测两种模式：
    1. 完整的工具调用 JSON（含 referenced_image_ids）
    2. 部分的工具参数 JSON（如 {"size":"1920x1088","n":1}）
    """
    if not message:
        return False
    if REFERENCED_IMAGE_IDS_RE.search(message):
        return True
    # 检测部分工具参数 JSON（模型返回了工具参数但未触发工具）
    if TOOL_PARAMS_JSON_RE.search(message):
        return True
    return False


def encode_images(images: Iterable[tuple[bytes, str, str]]) -> list[str]:
    return [base64.b64encode(data).decode("ascii") for data, _, _ in images if data]


def save_image_bytes(image_data: bytes, base_url: str | None = None) -> str:
    return image_storage_service.save(image_data, base_url).url


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def needs_global_system(messages: list[dict[str, Any]]) -> bool:
    """normalize_messages also runs inside conversation_events, on messages the protocol
    layer already normalized. Adding the global system prompt a second time sends the same
    system turn twice, which models tend to mirror back as duplicated instructions."""
    prompt = str(config.global_system_prompt or "")
    if not prompt:
        return False
    if not messages:
        return True
    first = messages[0]
    if str(first.get("role") or "") != "system":
        return True
    return message_text(first.get("content")) != prompt


def normalize_messages(messages: object, system: Any = None) -> list[dict[str, Any]]:
    source = [message for message in messages if isinstance(message, dict)] if isinstance(messages, list) else []
    normalized = []
    if needs_global_system(source):
        normalized.append({"role": "system", "content": config.global_system_prompt})
    system_text = message_text(system)
    if system_text:
        normalized.append({"role": "system", "content": system_text})
    for message in source:
        role = message.get("role", "user")
        content = message.get("content", "")
        text = message_text(content)
        images: list[tuple[bytes, str]] = []
        if role == "user":
            images.extend(extract_image_from_message_content(content))
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "image":
                        continue
                    data = part.get("data")
                    if isinstance(data, (bytes, bytearray)) and all(existing[0] != bytes(data) for existing in images):
                        images.append((bytes(data), str(part.get("mime") or "image/png")))
        if images:
            parts: list[Any] = []
            if text:
                parts.append({"type": "text", "text": text})
            for data, mime in images:
                parts.append({"type": "image", "data": data, "mime": mime})
            normalized.append({"role": role, "content": parts})
        else:
            normalized.append({"role": role, "content": text})
    return normalized


def prompt_with_global_system(prompt: str) -> str:
    return f"{config.global_system_prompt}\n\n{prompt}" if config.global_system_prompt else prompt


def assistant_history_text(messages: list[dict[str, Any]]) -> str:
    return "".join(str(item.get("content") or "") for item in messages if item.get("role") == "assistant")


def assistant_history_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("content") or "") for item in messages if item.get("role") == "assistant" and item.get("content")]


def build_image_prompt(prompt: str, size: str | None, quality: str = "auto") -> str:
    hints = []
    if size:
        hints.append(f"输出图片尺寸为 {size}。")
    if quality:
        hints.append(f"输出图片质量为 {quality}。")
    return f"{prompt.strip()}\n\n{''.join(hints)}" if hints else prompt


def encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def count_message_image_tokens(messages: list[dict[str, Any]], model: str) -> int:
    return sum(count_image_content_tokens(message.get("content"), model) for message in messages)


def count_message_text_tokens(messages: list[dict[str, Any]], model: str) -> int:
    encoding = encoding_for_model(model)
    total = 0
    for message in messages:
        total += 3
        for key, value in message.items():
            if key == "content" and isinstance(value, list):
                total += len(encoding.encode(message_text(value)))
            elif isinstance(value, str):
                total += len(encoding.encode(value))
            else:
                continue
            if key == "name":
                total += 1
    return total + 3


def count_message_tokens(messages: list[dict[str, Any]], model: str) -> int:
    return count_message_text_tokens(messages, model) + count_message_image_tokens(messages, model)


def count_text_tokens(text: str, model: str) -> int:
    return len(encoding_for_model(model).encode(text))


def format_image_result(
    items: list[dict[str, Any]],
    prompt: str,
    response_format: str,
    base_url: str | None = None,
    created: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for item in items:
        b64_json = str(item.get("b64_json") or "").strip()
        if not b64_json:
            continue
        revised_prompt = str(item.get("revised_prompt") or prompt).strip() or prompt
        if response_format == "b64_json":
            data.append({
                "b64_json": b64_json,
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
        else:
            data.append({
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if message and not data:
        result["message"] = message
    return result


@dataclass
class ConversationRequest:
    model: str = "auto"
    prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    thinking_effort: str = ""
    images: list[str] | None = None
    n: int = 1
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    base_url: str | None = None
    message_as_error: bool = False
    progress_callback: Any = None  # Callable[[str], None] | None


@dataclass
class ConversationState:
    # `text` follows what the client has actually received: deltas are append-only, so it
    # only ever grows, even when upstream rewrites what it is generating.
    text: str = ""
    conversation_id: str = ""
    file_ids: list[str] = field(default_factory=list)
    sediment_ids: list[str] = field(default_factory=list)
    blocked: bool = False
    tool_invoked: bool | None = None
    turn_use_case: str = ""


@dataclass
class MessageStream:
    """上游单条 assistant 消息的快照状态。

    同一条消息的后续事件只会把文本变长，所以「新增部分 = 新快照减去旧快照」是精确的；
    只有消息被改写、或者换了另一条消息时，才需要和已发文本做对齐。

    `shadow` 用于并行变体：上游可能为同一轮同时生成两条内容几乎相同的消息（见
    `variant_purpose`）。本轮一旦锁定主消息，其余 id 整条忽略，否则两份答案会交替追加
    进同一个输出里。
    """

    message_id: str = ""
    raw: str = ""
    text: str = ""
    tail_confirmed: bool = False
    shadow: bool = False
    finished: bool = False


@dataclass
class ImageOutput:
    kind: str
    model: str
    index: int
    total: int
    created: int = field(default_factory=lambda: int(time.time()))
    text: str = ""
    upstream_event_type: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)
    account_email: str = ""
    conversation_id: str = ""

    def to_chunk(self) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "object": "image.generation.chunk",
            "created": self.created,
            "model": self.model,
            "index": self.index,
            "total": self.total,
            "progress_text": self.text,
            "upstream_event_type": self.upstream_event_type,
            "data": [],
        }
        if self.account_email:
            chunk["_account_email"] = self.account_email
        if self.conversation_id:
            chunk["_conversation_id"] = self.conversation_id
        if self.kind == "message":
            chunk.update({
                "object": "image.generation.message",
                "message": self.text,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        elif self.kind == "result":
            chunk.update({
                "object": "image.generation.result",
                "data": self.data,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        return chunk


def assistant_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    parts = content.get("parts") or []
    if isinstance(parts, list) and parts:
        text = "".join(part for part in parts if isinstance(part, str))
        if text:
            return text
    # Fallback: content_type "code" stores text in the "text" field instead of "parts"
    text_field = str(content.get("text") or "")
    if text_field:
        return text_field
    return ""


def strip_history(text: str, history_text: str = "") -> str:
    text = str(text or "")
    history_text = str(history_text or "")
    while history_text and text.startswith(history_text):
        text = text[len(history_text):]
    return text


def common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


MIN_CONTAINED_LENGTH = 4
ANCHOR_PROBE_SIZES = (64, 32, 16, 8)
VARIANT_SWITCH_MARGIN = 32


def locate_snapshot_anchor(emitted: str, snapshot: str) -> int | None:
    """在已发文本里找快照开头的位置，用于对齐上游「重来」的那一段。"""
    for size in ANCHOR_PROBE_SIZES:
        if len(snapshot) < size:
            continue
        position = emitted.rfind(snapshot[:size])
        if position >= 0:
            return position
    return None


def emitted_prefix_length(emitted: str, snapshot: str) -> int:
    """How much of `snapshot`'s head the client has already received.

    Deltas are append-only, so anything the client already has must never be sent again.
    Upstream repeats text in three shapes, all of which show up as a snapshot that is not a
    plain extension of what was streamed:

    * the whole answer restarts from the greeting (regeneration / parallel variant),
    * the answer restarts at a section that was already streamed (`## 二、...`),
    * the earlier text is rewritten by sanitize_output_text, e.g. a collapsed space.

    The snapshot is compared with its leading whitespace removed: the first snapshot of a
    retake is often glued to the previous sentence while every later one starts with blank
    lines, and a byte-exact comparison made all of them look like brand new text. When the
    snapshot starts earlier than the end of the emitted text, the head is located inside it
    instead of only at the tail, which is what a section retake looks like.
    """
    if not snapshot:
        return 0
    stripped = snapshot.lstrip()
    if not stripped:
        # Only whitespace: let it through as the separator it is.
        return 0
    lead = len(snapshot) - len(stripped)
    body = stripped.rstrip()

    if len(body) >= MIN_CONTAINED_LENGTH and body in emitted:
        return len(snapshot)

    prefix = common_prefix_length(emitted, stripped)
    if prefix >= len(emitted) or prefix >= len(stripped):
        # 正常续写，或者快照本身就是已发文本的前缀。
        return min(len(snapshot), lead + prefix)

    # 上游从更早的位置重来：先在已发文本里定位快照开头，再按该位置算重叠长度。
    covered = prefix
    anchor = locate_snapshot_anchor(emitted, stripped)
    if anchor is not None:
        covered = max(covered, common_prefix_length(emitted[anchor:], stripped))
    if covered <= 0:
        return 0
    return lead + covered


def log_stream_join(state: ConversationState, stream: MessageStream, snapshot: str,
                    emitted: str, covered: int) -> None:
    """记录一次「新快照需要对已发文本做对齐」的拼接。

    单条消息从头流到尾只会在这里记一条（新消息的第一份快照）；如果同一条消息反复出现，
    说明上游在改写已经生成过的内容，日志里的 covered 就是两边的实际重合长度。
    """
    payload = {
        "event": "stream_text_join",
        "diverged": bool(stream.text),
        "message_id": stream.message_id,
        "conversation_id": state.conversation_id,
        "snapshot_length": len(snapshot),
        "emitted_length": len(emitted),
        "covered_length": covered,
        "appended_length": max(0, len(snapshot) - covered),
    }
    try:
        if stream.text:
            logger.info(payload)
        else:
            logger.debug(payload)
    except Exception:
        pass


def log_stream_switch(state: ConversationState, previous: MessageStream,
                      current: MessageStream, emitted_length: int) -> None:
    """记录一次「换一条消息继续输出」。

    `extend` 策略下，当前跟的那条停滞、另一条明显更长时切换过去；只补长度差额，
    所以既不会重复发送，也不会把后到的那份内容丢掉。
    """
    try:
        logger.info({
            "event": "stream_text_variant_switch",
            "conversation_id": state.conversation_id,
            "from_message_id": previous.message_id,
            "to_message_id": current.message_id,
            "from_length": len(previous.text),
            "to_length": len(current.text),
            "emitted_length": emitted_length,
        })
    except Exception:
        pass


def log_stream_variant(state: ConversationState, stream: MessageStream,
                       snapshot: str, emitted: str) -> None:
    """记录一条被忽略的并行变体。上游 `variant_purpose=comparison_implicit` 会同时生成两条。"""
    try:
        logger.info({
            "event": "stream_text_variant_shadowed",
            "message_id": stream.message_id,
            "conversation_id": state.conversation_id,
            "snapshot_length": len(snapshot),
            "emitted_length": len(emitted),
        })
    except Exception:
        pass


def sanitize_output_text(text: str) -> str:
    text = str(text or "")

    def is_internal_annotation_part(part: str) -> bool:
        value = part.strip()
        if not value:
            return True
        lower = value.lower()
        return bool(
            re.fullmatch(r"turn\d+[a-z]*\d*", lower)
            or re.fullmatch(r"turn\d+\w*", lower)
            or lower.startswith(("turn", "source", "sources"))
        )

    def readable_annotation_part(parts: list[str]) -> str:
        for part in parts:
            value = part.strip()
            if value and not is_internal_annotation_part(value):
                return value
        return ""

    def replace_annotation(match: re.Match[str]) -> str:
        payload = match.group(1)
        parts = [part.strip() for part in payload.split("\ue202")]
        kind = (parts[0] if parts else "").lower()
        data = parts[1:]
        if kind == "url":
            label = data[0] if data else ""
            url = data[1] if len(data) > 1 else ""
            if label and url.startswith(("http://", "https://")):
                return f"{label} ({url})"
            return label or url
        if kind == "cite":
            return readable_annotation_part(data)
        return readable_annotation_part(data)

    # ChatGPT web sometimes returns rich annotation markers using private-use
    # characters. API clients cannot render those. Preserve readable labels
    # from entity/link annotations, while removing internal citation pointers.
    text = re.sub(r"\ue200([^\ue201]*)\ue201", replace_annotation, text)
    text = re.sub(r"\ue200[^\ue201]*$", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    if config.strip_ui_markup:
        text = strip_ui_markup(text)
    return text


def assistant_raw_text(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if not isinstance(message, dict):
            continue
        role = str((message.get("author") or {}).get("role") or "").strip().lower()
        if role != "assistant":
            continue
        text = assistant_message_text(message)
        if text:
            return strip_history(text, history_text)
    return apply_text_patch(event, current_text, history_text)


def assistant_text(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    return sanitize_output_text(assistant_raw_text(event, current_text, history_text))


def event_assistant_text(event: dict[str, Any], history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
            return strip_history(assistant_message_text(message), history_text)
    return ""


def event_assistant_message_id(event: dict[str, Any]) -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if not isinstance(message, dict):
            continue
        if str((message.get("author") or {}).get("role") or "").strip().lower() != "assistant":
            continue
        return str(message.get("id") or message.get("message_id") or "")
    return ""


def finished_message_id(event: dict[str, Any]) -> str | None:
    """上游标记「这条消息写完了」时返回它的 id（空串表示归属当前消息），否则返回 None。

    上游写完一条消息后会长时间保持连接（并行变体时尤其明显），只等 `[DONE]` 会让客户端
    在答案已经完整之后还干等，所以这里把完成信号拿出来用于主动收尾。
    """
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if not isinstance(message, dict):
            continue
        if str((message.get("author") or {}).get("role") or "").strip().lower() != "assistant":
            continue
        if str(message.get("status") or "") == "finished_successfully" or message.get("end_turn") is True:
            return str(message.get("id") or message.get("message_id") or "")
    if event.get("p") == "/message/status" and event.get("v") == "finished_successfully":
        return ""
    return None


def apply_text_patch(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    if event.get("p") == "/message/content/parts/0":
        return apply_patch_op(event, current_text, history_text)

    operations = event.get("v")
    if isinstance(operations, str) and current_text and not event.get("p") and not event.get("o"):
        return current_text + operations

    if event.get("o") == "patch" and isinstance(operations, list):
        text = current_text
        for item in operations:
            if isinstance(item, dict):
                text = apply_text_patch(item, text, history_text)
        return text

    if not isinstance(operations, list):
        return current_text

    text = current_text
    for item in operations:
        if isinstance(item, dict):
            text = apply_text_patch(item, text, history_text)
    return text


def apply_patch_op(operation: dict[str, Any], current_text: str, history_text: str = "") -> str:
    op = operation.get("o")
    value = str(operation.get("v") or "")
    if op == "append":
        return current_text + value
    if op == "replace":
        return strip_history(value, history_text)
    return current_text


def add_unique(values: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in values:
            values.append(candidate)


FILE_SERVICE_ID_RE = re.compile(r"file-service://([A-Za-z0-9_-]+)")
FILE_ID_RE = re.compile(r"\b(file[-_](?!service\b)[A-Za-z0-9_-]+)\b")
# 真正的图片文件 ID 格式：file_00000000 + 24位十六进制字符（共32字符）
# 用于过滤非图片文件 ID（如 file_upload_business_upsell）
REAL_IMAGE_FILE_ID_RE = re.compile(r"\bfile_00000000[a-f0-9]{24}\b")
SEDIMENT_ID_RE = re.compile(r"sediment://([A-Za-z0-9_-]+)")


def extract_conversation_ids(payload: str) -> tuple[str, list[str], list[str]]:
    conversation_match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
    conversation_id = conversation_match.group(1) if conversation_match else ""
    file_ids: list[str] = []
    # Negative lookahead excludes "file-service" (URI prefix, not a real id).
    add_unique(file_ids, FILE_SERVICE_ID_RE.findall(payload))
    # 只提取真正的图片文件 ID（file_00000000... 格式），过滤非图片文件 ID（如 file_upload_business_upsell）
    add_unique(file_ids, REAL_IMAGE_FILE_ID_RE.findall(payload))
    sediment_ids = SEDIMENT_ID_RE.findall(payload)
    return conversation_id, file_ids, sediment_ids


def is_image_tool_event(event: dict[str, Any]) -> bool:
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata") or {}
    author = message.get("author") or {}
    content = message.get("content") or {}
    if author.get("role") != "tool":
        return False
    if metadata.get("async_task_type") == "image_gen":
        return True
    if content.get("content_type") != "multimodal_text":
        return False
    return any(
        isinstance(part, dict) and (
                part.get("content_type") == "image_asset_pointer"
                or str(part.get("asset_pointer") or "").startswith(("file-service://", "sediment://"))
        )
        for part in content.get("parts") or []
    )


def _is_user_message_event(event: dict[str, Any]) -> bool:
    """检查事件是否来自 user 角色消息。"""
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if isinstance(message, dict):
        author = message.get("author") or {}
        if str(author.get("role") or "").strip().lower() == "user":
            return True
    return False


def update_conversation_state(state: ConversationState, payload: str, event: dict[str, Any] | None = None) -> None:
    conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)
    if conversation_id and not state.conversation_id:
        state.conversation_id = conversation_id
    # Accept file_id / sediment_id when any of:
    #   1) event is a complete image_gen tool message
    #   2) prior server_ste_metadata already flipped tool_invoked True (in an image_gen turn),
    #      BUT only for non-user messages — user messages contain the uploaded input image
    #      which must NOT be treated as a generated output.
    #   3) patch event whose payload references asset_pointer / file-service://,
    #      BUT only when the event is not a user message.
    is_patch_event = isinstance(event, dict) and event.get("o") == "patch"
    is_user_msg = isinstance(event, dict) and _is_user_message_event(event)
    image_context = (
        (isinstance(event, dict) and is_image_tool_event(event))
        or (state.tool_invoked is True and not is_user_msg)
        or (is_patch_event and not is_user_msg and ("asset_pointer" in payload or "file-service://" in payload))
    )
    if image_context:
        add_unique(state.file_ids, file_ids)
        add_unique(state.sediment_ids, sediment_ids)
    if not isinstance(event, dict):
        return
    state.conversation_id = str(event.get("conversation_id") or state.conversation_id)
    value = event.get("v")
    if isinstance(value, dict):
        state.conversation_id = str(value.get("conversation_id") or state.conversation_id)
    if event.get("type") == "moderation":
        moderation = event.get("moderation_response")
        if isinstance(moderation, dict) and moderation.get("blocked") is True:
            state.blocked = True
    if event.get("type") == "server_ste_metadata":
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            if isinstance(metadata.get("tool_invoked"), bool):
                state.tool_invoked = metadata["tool_invoked"]
            state.turn_use_case = str(metadata.get("turn_use_case") or state.turn_use_case)


def conversation_base_event(event_type: str, state: ConversationState, **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "text": state.text,
        "conversation_id": state.conversation_id,
        "file_ids": list(state.file_ids),
        "sediment_ids": list(state.sediment_ids),
        "blocked": state.blocked,
        "tool_invoked": state.tool_invoked,
        "turn_use_case": state.turn_use_case,
        **extra,
    }


def iter_conversation_payloads(payloads: Iterator[str], history_text: str = "",
                               history_messages: list[str] | None = None,
                               stop_on_finish: bool = False) -> Iterator[dict[str, Any]]:
    state = ConversationState()
    history_messages = history_messages or []
    history_index = 0
    streams: dict[str, MessageStream] = {}
    current_message = ""
    primary_message: str | None = None
    leader_message: str | None = None
    for payload in payloads:
        # print(f"[upstream_sse] {payload}", flush=True)
        if not payload:
            continue
        if payload == "[DONE]":
            yield conversation_base_event("conversation.done", state, done=True)
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            update_conversation_state(state, payload)
            yield conversation_base_event("conversation.raw", state, payload=payload)
            continue
        if not isinstance(event, dict):
            yield conversation_base_event("conversation.event", state, raw=event)
            continue
        update_conversation_state(state, payload, event)
        if history_index < len(history_messages) and event_assistant_text(event, history_text) == history_messages[history_index]:
            history_index += 1
            streams.clear()
            current_message = ""
            primary_message = None
            leader_message = None
            state.text = ""
            continue
        message_id = event_assistant_message_id(event)
        if message_id:
            current_message = message_id
        stream_key = current_message
        stream = streams.get(stream_key)
        if stream is None:
            stream = MessageStream(message_id=stream_key)
            streams[stream_key] = stream
        next_raw_text = assistant_raw_text(event, stream.raw, history_text)
        next_text = sanitize_output_text(next_raw_text)
        stream.raw = next_raw_text
        finished_id = finished_message_id(event)
        if finished_id is not None:
            finished_stream = streams.get(finished_id or stream_key)
            if finished_stream is not None:
                finished_stream.finished = True

        delta = ""
        if not stream.shadow and next_text != stream.text:
            previous_text = stream.text
            stream.text = next_text
            emitted_text = state.text
            policy = config.stream_variant_policy
            skip = False
            if policy == "follow_first" and primary_message is not None and stream_key != primary_message:
                # 本轮已锁定一条消息，其他 id 是并行变体：整条忽略。
                stream.shadow = True
                log_stream_variant(state, stream, next_text, emitted_text)
                skip = True
            elif policy == "extend" and leader_message is not None and stream_key != leader_message:
                leader = streams.get(leader_message)
                if leader is None or len(next_text) <= len(leader.text) + VARIANT_SWITCH_MARGIN:
                    # 明显落后于当前跟的那条，先不发；等它真的更长再切过去。
                    skip = True
                else:
                    log_stream_switch(state, leader, stream, len(emitted_text))
                    leader_message = stream_key

            if not skip:
                if policy == "extend":
                    # 只补「已发长度」之后的差额：并行两条消息时既不会重复，也不会把后到的那份丢掉。
                    delta = next_text[len(emitted_text):] if len(next_text) > len(emitted_text) else ""
                elif stream.tail_confirmed and next_text.startswith(previous_text):
                    # 同一条消息在变长，而且客户端当前的结尾正是这条消息上一次的快照：
                    # 快照之间的前缀关系是协议保证的，直接取差额即可。
                    delta = next_text[len(previous_text):]
                else:
                    # 换了消息、这条消息被改写、或者上一次的内容并不是由这条消息追加的：
                    # 只发客户端还没有的部分。
                    covered = emitted_prefix_length(emitted_text, next_text)
                    delta = next_text[covered:]
                    log_stream_join(state, stream, next_text, emitted_text, covered)

                if delta:
                    if policy == "extend" and leader_message is None:
                        leader_message = stream_key
                    if policy == "follow_first" and primary_message is None:
                        # 第一条真正产出内容的消息就是本轮的主消息。
                        primary_message = stream_key
                    state.text = emitted_text + delta
                stream.tail_confirmed = state.text.endswith(next_text)

        if delta:
            yield conversation_base_event("conversation.delta", state, raw=event, delta=delta)
        else:
            yield conversation_base_event("conversation.event", state, raw=event)

        if stop_on_finish and streams and all(
            item.finished for item in streams.values() if not item.shadow
        ):
            # 上游已经把这条（或这些）消息标记为完成，不必再等它关连接。
            yield conversation_base_event("conversation.done", state, done=True)
            break


def conversation_events(
    backend: OpenAIBackendAPI,
    messages: list[dict[str, Any]] | None = None,
    model: str = "auto",
    prompt: str = "",
    images: list[str] | None = None,
    size: str | None = None,
    quality: str = "auto",
    thinking_effort: str = "",
) -> Iterator[dict[str, Any]]:
    normalized = normalize_messages(messages or ([{"role": "user", "content": prompt}] if prompt else []))
    image_model = is_supported_image_model(model)
    history_text = "" if image_model else assistant_history_text(normalized)
    history_messages = [] if image_model else assistant_history_messages(normalized)
    final_prompt = prompt_with_global_system(build_image_prompt(prompt, size, quality)) if image_model else prompt
    payloads = backend.stream_conversation(
        messages=normalized,
        model=model,
        prompt=final_prompt,
        images=images if image_model else None,
        system_hints=["picture_v2"] if image_model else None,
        thinking_effort=thinking_effort if not image_model else "",
    )
    yield from iter_conversation_payloads(
        payloads, history_text, history_messages, stop_on_finish=not image_model
    )


def text_backend() -> OpenAIBackendAPI:
    return OpenAIBackendAPI(access_token=account_service.get_text_access_token())


def stream_text_deltas(backend: OpenAIBackendAPI, request: ConversationRequest) -> Iterator[str]:
    attempted_tokens: set[str] = set()
    token = getattr(backend, "access_token", "")
    emitted = False
    started_at = time.monotonic()
    last_delta_at = 0.0
    delta_count = 0
    had_deadline = hasattr(backend, "request_deadline")
    previous_deadline = getattr(backend, "request_deadline", None)
    try:
        request_deadline = float(previous_deadline) if previous_deadline is not None else time.monotonic() + TEXT_STREAM_TIMEOUT_SECS
    except (TypeError, ValueError):
        request_deadline = time.monotonic() + TEXT_STREAM_TIMEOUT_SECS
    backend.request_deadline = request_deadline
    try:
        while True:
            if token and token in attempted_tokens:
                raise RuntimeError("no available text account")
            if token:
                attempted_tokens.add(token)
            active_backend = None
            try:
                active_backend = OpenAIBackendAPI(access_token=token)
                active_backend.request_deadline = request_deadline
                for event in conversation_events(
                    active_backend,
                    messages=request.messages,
                    model=request.model,
                    prompt=request.prompt,
                    thinking_effort=request.thinking_effort,
                ):
                    if event.get("type") != "conversation.delta":
                        continue
                    delta = str(event.get("delta") or "")
                    if delta:
                        emitted = True
                        delta_count += 1
                        last_delta_at = time.monotonic()
                        yield delta
                account_service.mark_text_used(token)
                return
            except Exception as exc:
                error_message = str(exc)
                if token and not emitted and is_token_invalid_error(error_message):
                    refreshed_token = account_service.refresh_access_token(token, force=True, event="text_stream")
                    if refreshed_token and refreshed_token != token and refreshed_token not in attempted_tokens:
                        token = refreshed_token
                    else:
                        account_service.remove_invalid_token(token, "text_stream")
                        token = account_service.get_text_access_token(attempted_tokens)
                    if token:
                        continue
                raise
            finally:
                if active_backend is not None:
                    active_backend.close()
    finally:
        if emitted:
            # `tail_gap_ms` 是「最后一次吐出文本」到「流真正结束」之间的间隔，
            # 客户端感觉到的"卡着"就是这个值。
            try:
                logger.info({
                    "event": "text_stream_metrics",
                    "deltas": delta_count,
                    "total_ms": int((time.monotonic() - started_at) * 1000),
                    "tail_gap_ms": int((time.monotonic() - last_delta_at) * 1000) if last_delta_at else 0,
                })
            except Exception:
                pass
        if had_deadline:
            backend.request_deadline = previous_deadline
        elif hasattr(backend, "request_deadline"):
            delattr(backend, "request_deadline")


def collect_text(backend: OpenAIBackendAPI, request: ConversationRequest) -> str:
    return "".join(stream_text_deltas(backend, request))


def _get_detailed_error_from_tasks(
    backend: OpenAIBackendAPI,
    conversation_id: str,
    timeout_secs: float = 10.0,
    wait_secs: float = 2.0,
) -> str:
    """从 /backend-api/tasks/ 接口获取结构化错误信息。

    当 SSE 流检测到 moderation 拦截时，轮询 tasks 接口获取详细错误文本。
    使用结构化字段（metadata.is_error, author.role, content.content_type）判断，
    而非依赖易变的文本匹配。

    参数：
    - `backend`：OpenAIBackendAPI 实例。
    - `conversation_id`：会话 ID。
    - `timeout_secs`：请求超时秒数。
    - `wait_secs`：等待任务创建的秒数。设为 0 可跳过等待。

    返回：
    - 详细错误信息文本，如果未找到则返回空字符串。
    """
    import time as _time
    try:
        if wait_secs > 0:
            _time.sleep(wait_secs)
        tasks = backend._query_backend_tasks(conversation_id=conversation_id, timeout_secs=timeout_secs)
        if not tasks:
            return ""

        for task in tasks:
            is_error, error_msg, metadata = backend.check_task_error(task)
            if is_error and error_msg:
                logger.info({
                    "event": "image_task_structured_error",
                    "conversation_id": conversation_id,
                    "error_msg": error_msg,
                    "metadata": metadata,
                })
                return error_msg
        return ""
    except Exception as exc:
        logger.warning({
            "event": "image_task_error_query_failed",
            "conversation_id": conversation_id,
            "error": str(exc),
        })
        return ""


def _remove_image_conversation_later(backend: OpenAIBackendAPI, conversation_id: str) -> None:
    if not config.image_remove_conversation_after_result or not conversation_id:
        return

    def _run() -> None:
        try:
            backend.delete_conversation(conversation_id)
            logger.info({"event": "image_conversation_removed", "conversation_id": conversation_id})
        except Exception as exc:
            logger.warning({
                "event": "image_conversation_remove_failed",
                "conversation_id": conversation_id,
                "error": str(exc),
            })

    threading.Thread(target=_run, name=f"remove-image-conversation-{conversation_id}", daemon=True).start()


def stream_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        index: int = 1,
        total: int = 1,
) -> Iterator[ImageOutput]:
    last: dict[str, Any] = {}
    for event in conversation_events(
            backend,
            prompt=request.prompt,
            model=request.model,
            images=request.images or [],
            size=request.size,
            quality=request.quality,
    ):
        last = event
        if event.get("type") == "conversation.delta":
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                text=str(event.get("delta") or ""),
                upstream_event_type="conversation.delta",
            )
            continue
        if event.get("type") == "conversation.event":
            raw = event.get("raw")
            raw_type = str(raw.get("type") or "") if isinstance(raw, dict) else ""
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                upstream_event_type=raw_type,
            )

    conversation_id = str(last.get("conversation_id") or "")
    file_ids = [str(item) for item in last.get("file_ids") or []]
    sediment_ids = [str(item) for item in last.get("sediment_ids") or []]
    message = str(last.get("text") or "").strip()
    is_text_reply = bool(message and is_model_text_reply_instead_of_image(message))
    logger.info({
        "event": "image_stream_resolve_start",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "tool_invoked": last.get("tool_invoked"),
        "turn_use_case": last.get("turn_use_case"),
    })
    if request.progress_callback:
        request.progress_callback("image_stream_resolve_start")
    if message and not file_ids and not sediment_ids and last.get("blocked"):
        # 尝试从 /backend-api/tasks/ 获取详细错误信息
        detailed_error = _get_detailed_error_from_tasks(backend, conversation_id)
        error_text = detailed_error or message or "Image generation was rejected by upstream policy."
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=error_text, conversation_id=conversation_id)
        return
    should_poll_for_image = bool(request.images) or last.get("turn_use_case") == "image gen"
    if message and not file_ids and not sediment_ids and not should_poll_for_image and not is_text_reply:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message, conversation_id=conversation_id)
        return

    # 检测模型是否返回了文本描述（含 referenced_image_ids）而非实际生成图片
    # 这说明模型已发起图片生成工具调用，但 SSE 在工具完成前断开，
    # 图片可能正在异步生成中。需要使用更积极的轮询策略来获取结果。
    if is_text_reply:
        logger.info({
            "event": "image_detected_text_reply_with_ids",
            "conversation_id": conversation_id,
            "message_preview": message[:200],
        })

    # 当检测到文本回复但 conversation_id 丢失时，尝试从最近对话列表中恢复
    # SSE 流太短时（模型返回文本而非触发图片工具），conversation_id 可能未被捕获，
    # 但图片已在上游异步生成。通过列出最近对话来恢复 conversation_id。
    if is_text_reply and not conversation_id:
        try:
            import time as _time
            recovered_id = backend.find_conversation_by_prompt(
                request.prompt, _time.time(), timeout_secs=5.0,
            )
            if recovered_id:
                conversation_id = recovered_id
                logger.info({
                    "event": "image_conversation_id_recovered",
                    "conversation_id": conversation_id,
                    "message_preview": message[:200],
                })
        except Exception as exc:
            logger.warning({
                "event": "image_conversation_id_recovery_failed",
                "error": repr(exc)[:300],
            })

    # 在轮询图片之前，先检查 /backend-api/tasks/ 是否有 moderation 拦截
    # 这样可以避免不必要的长时间轮询超时
    # 注意：当 should_poll_for_image 为 True 或检测到文本回复时，
    # 即使 tasks 报告了"错误"，也不能直接返回——因为上游可能将工具调用的 JSON 参数
    # （如 {"size":"1792x1024","n":1}）标记为 is_error，而实际上图片正在异步生成中。
    # 此时应继续轮询图片。
    detailed_error = ""
    if not file_ids and not sediment_ids and conversation_id:
        detailed_error = _get_detailed_error_from_tasks(backend, conversation_id, timeout_secs=5.0, wait_secs=1.0)
        if detailed_error and not should_poll_for_image and not is_text_reply:
            logger.info({
                "event": "image_task_error_before_poll",
                "conversation_id": conversation_id,
                "error": detailed_error,
            })
            yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=detailed_error, conversation_id=conversation_id)
            return
        if detailed_error and (should_poll_for_image or is_text_reply):
            logger.info({
                "event": "image_task_error_skipped_for_poll",
                "conversation_id": conversation_id,
                "error": detailed_error,
            })

    # 当检测到文本回复（含 referenced_image_ids）时，使用更长的超时来轮询图片结果。
    # 因为上游可能将图片生成作为异步任务执行，SSE 流在工具完成前就断开了，
    # 导致对话文档中尚未写入图片工具的响应记录。
    poll_timeout = _image_remaining_timeout_secs(backend)
    if is_text_reply and conversation_id:
        logger.info({
            "event": "image_text_reply_extended_poll",
            "conversation_id": conversation_id,
            "poll_timeout_secs": poll_timeout,
        })

    try:
        image_urls = backend.resolve_conversation_image_urls(
            conversation_id, file_ids, sediment_ids, poll_timeout_secs=poll_timeout,
        )
    except (ImageContentPolicyError, ImagePollTimeoutError) as exc:
        # 当检测到文本回复时，task error 不应直接判定为内容策略违规，
        # 因为图片可能仍在后台异步生成中
        if is_text_reply and isinstance(exc, ImageContentPolicyError):
            logger.warning({
                "event": "image_text_reply_task_error_ignored",
                "conversation_id": conversation_id,
                "error": str(exc),
            })
            image_urls = []
        else:
            raise
    except Exception as exc:
        # 当检测到文本回复时，首次轮询的临时网络错误不应直接中断，
        # 因为图片可能仍在后台异步生成中，后续 retry poll 会继续尝试。
        if is_text_reply and conversation_id:
            logger.warning({
                "event": "image_text_reply_first_poll_error_ignored",
                "conversation_id": conversation_id,
                "error": repr(exc)[:300],
            })
            image_urls = []
        else:
            raise

    if image_urls:
        if request.progress_callback:
            request.progress_callback("receiving_image")
        image_items = [
            {"b64_json": base64.b64encode(image_data).decode("ascii")}
            for image_data in backend.download_image_bytes(image_urls)
        ]
        data = format_image_result(
            image_items,
            request.prompt,
            request.response_format,
            request.base_url,
            int(time.time()),
        )["data"]
        if data:
            _remove_image_conversation_later(backend, conversation_id)
            yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data, conversation_id=conversation_id)
        return

    if message:
        # 检测模型是否返回了文本描述（含 referenced_image_ids）而非实际生成图片
        # 这说明模型已发起图片生成工具调用，但 SSE 在工具完成前断开。
        # 此时应再尝试轮询图片结果，而不是直接把文本当作最终输出。
        # 当 is_text_reply 但 conversation_id 丢失时，尝试从最近对话列表恢复
        if is_text_reply and not conversation_id:
            try:
                import time as _time
                recovered_id = backend.find_conversation_by_prompt(
                    request.prompt, _time.time(), timeout_secs=5.0,
                )
                if recovered_id:
                    conversation_id = recovered_id
                    logger.info({
                        "event": "image_text_reply_conversation_id_recovered",
                        "conversation_id": conversation_id,
                        "message_preview": message[:200],
                    })
            except Exception as exc:
                logger.warning({
                    "event": "image_text_reply_conversation_id_recovery_failed",
                    "error": repr(exc)[:300],
                })
        if is_text_reply and conversation_id:
            logger.info({
                "event": "image_model_text_reply_retry_poll",
                "conversation_id": conversation_id,
                "message_preview": message[:200],
            })
            # 文本回复场景下，图片可能需要 4-5 分钟才能异步生成完成。
            # 使用配置的剩余总预算并允许短重试，避免因临时网络问题提前退出。
            MAX_POLL_RETRIES = 3
            for poll_attempt in range(1, MAX_POLL_RETRIES + 1):
                retry_poll_timeout = _image_remaining_timeout_secs(backend)
                if retry_poll_timeout <= 0:
                    break
                try:
                    polled_file_ids, polled_sediment_ids = backend._poll_image_results(
                        conversation_id,
                        retry_poll_timeout,
                        file_ids,
                        sediment_ids,
                    )
                    file_ids.extend(item for item in polled_file_ids if item and item not in file_ids)
                    sediment_ids.extend(item for item in polled_sediment_ids if item and item not in sediment_ids)
                    break  # 轮询成功，退出重试循环
                except Exception as exc:
                    error_str = str(exc)
                    is_transient = (
                        isinstance(exc, ImagePollTimeoutError)
                        or is_tls_connection_error(error_str)
                        or "upstream" in error_str.lower()
                        or "connection" in error_str.lower()
                        or "timeout" in error_str.lower()
                    )
                    logger.warning({
                        "event": "image_model_text_reply_poll_failed",
                        "conversation_id": conversation_id,
                        "poll_attempt": poll_attempt,
                        "error": repr(exc)[:300],
                        "is_transient": is_transient,
                    })
                    # 如果还有重试次数且不是超时/内容违规错误，继续重试
                    if poll_attempt < MAX_POLL_RETRIES and not isinstance(exc, (ImagePollTimeoutError, ImageContentPolicyError)):
                        # 递增退避：30s, 60s, 90s
                        backoff = 30.0 * poll_attempt
                        logger.info({
                            "event": "image_model_text_reply_poll_retry",
                            "conversation_id": conversation_id,
                            "poll_attempt": poll_attempt,
                            "backoff_secs": backoff,
                        })
                        _sleep_image_retry(backend, backoff)
                        continue
                    # 超时错误或重试次数用尽，停止重试
                    break

            if file_ids or sediment_ids:
                image_urls = backend.resolve_conversation_image_urls(
                    conversation_id, file_ids, sediment_ids, poll=False,
                )
                if image_urls:
                    if request.progress_callback:
                        request.progress_callback("receiving_image")
                    image_items = [
                        {"b64_json": base64.b64encode(image_data).decode("ascii")}
                        for image_data in backend.download_image_bytes(image_urls)
                    ]
                    data = format_image_result(
                        image_items,
                        request.prompt,
                        request.response_format,
                        request.base_url,
                        int(time.time()),
                    )["data"]
                    if data:
                        _remove_image_conversation_later(backend, conversation_id)
                        yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data, conversation_id=conversation_id)
                        return
        elif is_text_reply:
            logger.warning({
                "event": "image_model_text_reply_no_image",
                "conversation_id": conversation_id,
                "message_preview": message[:200],
            })
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message, conversation_id=conversation_id)
        return

    # 兜底：当 message 为空且图片 URL 解析失败时，先尝试一次短延迟重试轮询
    # 然后抛出明确错误而非让调用方得到 "upstream completed without generating images" 这种模糊报错
    logger.warning({
        "event": "image_stream_no_result_fallback",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "should_poll_for_image": should_poll_for_image,
    })
    # 当 should_poll_for_image 为 True 但 conversation_id 丢失时，尝试恢复
    if should_poll_for_image and not conversation_id:
        try:
            import time as _time
            recovered_id = backend.find_conversation_by_prompt(
                request.prompt, _time.time(), timeout_secs=5.0,
            )
            if recovered_id:
                conversation_id = recovered_id
                logger.info({
                    "event": "image_fallback_conversation_id_recovered",
                    "conversation_id": conversation_id,
                })
        except Exception as exc:
            logger.warning({
                "event": "image_fallback_conversation_id_recovery_failed",
                "error": repr(exc)[:300],
            })
    if should_poll_for_image and conversation_id:
        # 图片可能仍在异步处理中（上游 SSE 流在图片生成完成前就结束了）。
        # 使用配置的剩余总预算并允许短重试，避免因临时网络问题或图片尚未提交而提前退出。
        MAX_FALLBACK_POLL_RETRIES = 3
        for poll_attempt in range(1, MAX_FALLBACK_POLL_RETRIES + 1):
            retry_wait_secs = min(30.0 * poll_attempt, config.image_poll_initial_wait_secs * poll_attempt)
            logger.info({
                "event": "image_stream_retry_poll_after_wait",
                "conversation_id": conversation_id,
                "retry_wait_secs": retry_wait_secs,
                "poll_attempt": poll_attempt,
            })
            _sleep_image_retry(backend, retry_wait_secs)
            retry_poll_timeout = _image_remaining_timeout_secs(backend)
            if retry_poll_timeout <= 0:
                break
            try:
                polled_file_ids, polled_sediment_ids = backend._poll_image_results(
                    conversation_id,
                    retry_poll_timeout,
                    file_ids,
                    sediment_ids,
                )
                file_ids.extend(item for item in polled_file_ids if item and item not in file_ids)
                sediment_ids.extend(item for item in polled_sediment_ids if item and item not in sediment_ids)
                break  # 轮询成功，退出重试循环
            except Exception as exc:
                error_str = str(exc)
                is_transient = (
                    isinstance(exc, ImagePollTimeoutError)
                    or is_tls_connection_error(error_str)
                    or "upstream" in error_str.lower()
                    or "connection" in error_str.lower()
                    or "timeout" in error_str.lower()
                )
                logger.warning({
                    "event": "image_stream_retry_poll_failed",
                    "conversation_id": conversation_id,
                    "poll_attempt": poll_attempt,
                    "error": repr(exc)[:300],
                    "is_transient": is_transient,
                })
                # 如果还有重试次数且不是超时/内容违规错误，继续重试
                if poll_attempt < MAX_FALLBACK_POLL_RETRIES and not isinstance(exc, (ImagePollTimeoutError, ImageContentPolicyError)):
                    # 递增退避：30s, 60s
                    backoff = 30.0 * poll_attempt
                    logger.info({
                        "event": "image_stream_retry_poll_retry",
                        "conversation_id": conversation_id,
                        "poll_attempt": poll_attempt,
                        "backoff_secs": backoff,
                    })
                    _sleep_image_retry(backend, backoff)
                    continue
                # 超时错误或重试次数用尽，停止重试
                break
        
        if file_ids or sediment_ids:
            image_urls = backend.resolve_conversation_image_urls(
                conversation_id, file_ids, sediment_ids, poll=False,
            )
            if image_urls:
                if request.progress_callback:
                    request.progress_callback("receiving_image")
                image_items = [
                    {"b64_json": base64.b64encode(image_data).decode("ascii")}
                    for image_data in backend.download_image_bytes(image_urls)
                ]
                data = format_image_result(
                    image_items,
                    request.prompt,
                    request.response_format,
                    request.base_url,
                    int(time.time()),
                )["data"]
                if data:
                    _remove_image_conversation_later(backend, conversation_id)
                    yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data, conversation_id=conversation_id)
                    return
        
        # 重试后仍然失败，yield 错误消息
        yield ImageOutput(kind="message", model=request.model, index=index, total=total,
                          text="Image generation completed upstream but the result could not be retrieved. "
                               "The image may still be processing. Please try again in a moment.",
                          conversation_id=conversation_id)
    elif message:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message, conversation_id=conversation_id)
    else:
        # conversation_id 也为空时（SSE 流极短、未捕获到会话 ID），
        # 仍然 yield 一条消息，避免 stream_image_outputs_with_pool 产生
        # "upstream completed without generating images" 模糊报错
        yield ImageOutput(kind="message", model=request.model, index=index, total=total,
                          text="Image generation started upstream but the response was incomplete. "
                               "Please try again.",
                          conversation_id=conversation_id)


def _codex_response_images(value: Any) -> list[str]:
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call" and isinstance(value.get("result"), str):
            result = value["result"].strip()
            if result:
                return [result.split(",", 1)[1] if result.startswith("data:image/") else result]
        images: list[str] = []
        for item in value.values():
            images.extend(_codex_response_images(item))
        return images
    if isinstance(value, list):
        images: list[str] = []
        for item in value:
            images.extend(_codex_response_images(item))
        return images
    return []


def stream_codex_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        index: int = 1,
        total: int = 1,
) -> Iterator[ImageOutput]:
    images = _codex_response_images(list(backend.iter_codex_image_response_events(
        prompt=request.prompt,
        images=request.images or [],
        size=request.size,
        quality=request.quality,
    )))
    if not images:
        raise ImageGenerationError("No image result found in response")
    data = format_image_result(
        [{"b64_json": item, "revised_prompt": request.prompt} for item in images],
        request.prompt,
        request.response_format,
        request.base_url,
        int(time.time()),
    )["data"]
    if data:
        yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data)
        return
    raise ImageGenerationError("No image result found in response")


def _generate_single_image(
        request: ConversationRequest,
        index: int,
        total: int,
) -> list[ImageOutput]:
    """为单张图片执行生成逻辑（含重试），返回结果列表。

    该函数在独立线程中运行，每个线程使用不同的账号，
    实现并行生图，避免串行超时阻塞。
    """
    # 模型返回文本而非图片的最大重试次数
    MAX_TEXT_REPLY_RETRIES = 3
    # TLS 连接错误最大重试次数
    MAX_TLS_RETRIES = 3
    # 连接超时错误最大重试次数（同账号短等待重试）
    MAX_CONN_TIMEOUT_RETRIES = 3
    # 轮询超时错误最大重试次数（换账号重试）
    MAX_POLL_TIMEOUT_RETRIES = 4
    # 文件服务并发限流最大重试次数（换账号重试）
    MAX_CONCURRENCY_RETRIES = 3

    text_reply_retry_count = 0
    tls_retry_count = 0
    conn_timeout_retry_count = 0
    poll_timeout_retry_count = 0
    concurrency_retry_count = 0
    account_email = ""
    request_deadline = time.monotonic() + float(config.image_poll_timeout_secs)

    while True:
        remaining_before_attempt = request_deadline - time.monotonic()
        if remaining_before_attempt <= 0:
            raise ImagePollTimeoutError(
                f"ChatGPT 生图超时（已等待 {config.image_poll_timeout_secs} 秒）。"
                f"当前超时阈值可在 config.json 中调大 image_poll_timeout_secs，"
                f"也可能是账号被限流或生图队列拥堵导致。"
            )
        try:
            if request.progress_callback:
                request.progress_callback("getting_account")
            plan_type, _ = split_image_model(request.model)
            codex_model = is_codex_image_model(request.model)
            token = account_service.get_available_access_token(
                plan_type=plan_type,
                source_type="codex" if codex_model else None,
                plan_types=("plus", "team", "pro") if codex_model and not plan_type else None,
            )
        except RuntimeError as exc:
            raise ImageGenerationError(str(exc) or "image generation failed", account_email=account_email) from exc

        emitted_for_token = False
        returned_message = False
        returned_result = False
        account = account_service.get_account(token) or {}
        account_email = str(account.get("email") or "").strip()
        logger.debug({
            "event": "image_account_lookup",
            "token_prefix": token[:12] + "..." if len(token) > 12 else token,
            "account_email": account_email,
            "account_found": bool(account),
            "index": index,
        })
        backend = None
        try:
            backend = OpenAIBackendAPI(access_token=token)
            backend.image_request_deadline = request_deadline
            if request.progress_callback:
                backend.progress_callback = request.progress_callback
            stream_fn = stream_codex_image_outputs if is_codex_image_model(request.model) else stream_image_outputs
            outputs: list[ImageOutput] = []
            for output in stream_fn(backend, request, index, total):
                if account_email and not output.account_email:
                    output.account_email = account_email
                if output.kind == "message" and request.message_as_error:
                    raise ImageGenerationError(
                        output.text or "Image generation was rejected by upstream policy.",
                        status_code=400,
                        error_type="invalid_request_error",
                        code="content_policy_violation",
                        account_email=account_email,
                        conversation_id=output.conversation_id,
                    )
                emitted_for_token = True
                returned_message = output.kind == "message"
                returned_result = returned_result or output.kind == "result"
                outputs.append(output)
            if returned_message:
                account_service.mark_image_result(token, False)
                return outputs
            if not returned_result:
                account_service.mark_image_result(token, False)
                if emitted_for_token:
                    conv_id = outputs[-1].conversation_id if outputs else ""
                    raise ImageGenerationError(
                        "upstream completed without generating images",
                        status_code=400,
                        error_type="invalid_request_error",
                        code="no_image_generated",
                        account_email=account_email,
                        conversation_id=conv_id,
                    )
                return outputs
            account_service.mark_image_result(token, True)
            return outputs
        except ImagePollTimeoutError as exc:
            account_service.mark_image_result(token, False)
            if account_email:
                setattr(exc, "account_email", account_email)
            # 轮询超时：换账号重试
            if not emitted_for_token:
                poll_timeout_retry_count += 1
                if poll_timeout_retry_count <= MAX_POLL_TIMEOUT_RETRIES and request_deadline > time.monotonic():
                    logger.warning({
                        "event": "image_poll_timeout_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": poll_timeout_retry_count,
                        "index": index,
                        "error": str(exc)[:200],
                    })
                    continue
                logger.warning({
                    "event": "image_poll_timeout_exhausted_retries",
                    "request_token": token,
                    "account_email": account_email,
                    "retry_count": poll_timeout_retry_count,
                    "index": index,
                })
                raise
            raise
        except ImageContentPolicyError as exc:
            account_service.mark_image_result(token, False)
            logger.warning({
                "event": "image_stream_content_policy_error",
                "request_token": token,
                "account_email": account_email,
                "error": str(exc),
                "index": index,
            })
            raise ImageGenerationError(
                str(exc) or "Image generation was rejected by upstream policy.",
                status_code=400,
                error_type="invalid_request_error",
                code="content_policy_violation",
                account_email=account_email,
                conversation_id=getattr(exc, "conversation_id", ""),
            ) from exc
        except ImageGenerationError as exc:
            account_service.mark_image_result(token, False)
            if account_email and not getattr(exc, "account_email", ""):
                exc.account_email = account_email
            error_text = str(exc)
            # 如果是模型返回文本而非图片，尝试换账号重试
            if is_model_text_reply_instead_of_image(error_text) and not emitted_for_token:
                text_reply_retry_count += 1
                if text_reply_retry_count <= MAX_TEXT_REPLY_RETRIES:
                    logger.warning({
                        "event": "image_model_text_reply_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": text_reply_retry_count,
                        "index": index,
                        "error": error_text[:200],
                    })
                    continue
                logger.warning({
                    "event": "image_model_text_reply_exhausted_retries",
                    "request_token": token,
                    "account_email": account_email,
                    "retry_count": text_reply_retry_count,
                    "index": index,
                })
                raise ImageGenerationError(
                    "Image generation failed: the upstream model returned a text description "
                    "instead of generating an image. Please try again later.",
                    status_code=502,
                    error_type="server_error",
                    code="upstream_text_reply",
                    account_email=account_email,
                    conversation_id=getattr(exc, "conversation_id", ""),
                ) from exc
            logger.warning({
                "event": "image_stream_generation_error",
                "request_token": token,
                "account_email": account_email,
                "error": error_text,
                "index": index,
            })
            raise
        except Exception as exc:
            cooldown_secs = float(config.image_timeout_retry_secs) if (
                isinstance(exc, ImageStreamHardTimeoutError) or is_image_concurrency_error(exc)
            ) else 0.0
            account_service.mark_image_result(token, False, cooldown_secs=cooldown_secs)
            last_error = str(exc)
            logger.warning({
                "event": "image_stream_fail",
                "request_token": token,
                "account_email": account_email,
                "error": last_error,
                "index": index,
            })
            if not emitted_for_token and is_image_concurrency_error(exc):
                concurrency_retry_count += 1
                if concurrency_retry_count <= MAX_CONCURRENCY_RETRIES and request_deadline > time.monotonic():
                    wait_secs = min(float(config.image_timeout_retry_secs), 10.0)
                    logger.warning({
                        "event": "image_stream_concurrency_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": concurrency_retry_count,
                        "index": index,
                        "wait_secs": wait_secs,
                        "error": last_error[:300],
                    })
                    _sleep_until_deadline(request_deadline, wait_secs)
                    continue
            if not emitted_for_token and is_token_invalid_error(last_error):
                refreshed_token = account_service.refresh_access_token(token, force=True, event="image_stream")
                if refreshed_token and refreshed_token != token:
                    token = refreshed_token
                    continue
                account_service.remove_invalid_token(token, "image_stream")
                continue
            # TLS/SSL 连接错误：自动重试
            if not emitted_for_token and is_tls_connection_error(last_error):
                tls_retry_count += 1
                if tls_retry_count <= MAX_TLS_RETRIES and request_deadline > time.monotonic():
                    wait_secs = min(2.0 * tls_retry_count, 10.0)
                    logger.warning({
                        "event": "image_stream_tls_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": tls_retry_count,
                        "index": index,
                        "wait_secs": wait_secs,
                        "error": last_error[:200],
                    })
                    _sleep_until_deadline(request_deadline, wait_secs)
                    continue
            # 连接超时错误（curl 28）：同账号短等待重试，不切换账号
            if not emitted_for_token and is_connection_timeout_error(last_error):
                conn_timeout_retry_count += 1
                if conn_timeout_retry_count <= MAX_CONN_TIMEOUT_RETRIES and request_deadline > time.monotonic():
                    wait_secs = min(3.0 * conn_timeout_retry_count, 9.0)
                    logger.warning({
                        "event": "image_stream_conn_timeout_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": conn_timeout_retry_count,
                        "index": index,
                        "wait_secs": wait_secs,
                        "error": last_error[:200],
                    })
                    _sleep_until_deadline(request_deadline, wait_secs)
                    continue
            raise ImageGenerationError(image_stream_error_message(last_error), account_email=account_email, conversation_id="") from exc
        finally:
            if backend is not None:
                backend.close()


def stream_image_outputs_with_pool(request: ConversationRequest) -> Iterator[ImageOutput]:
    """并行生成多张图片，每张图片使用独立线程和账号，互不阻塞。"""
    if not is_supported_image_model(request.model):
        raise ImageGenerationError("unsupported image model,supported models: " + ", ".join(sorted(supported_image_models())))

    if request.n <= 1:
        # 单张图片，直接执行（无需线程池开销）
        outputs = _generate_single_image(request, 1, 1)
        for output in outputs:
            yield output
        return

    # 多张图片：根据配置选择并行或串行执行
    if not config.image_parallel_generation:
        logger.info({
            "event": "image_serial_generation_start",
            "n": request.n,
            "model": request.model,
        })
        for index in range(1, request.n + 1):
            outputs = _generate_single_image(request, index, request.n)
            for output in outputs:
                yield output
        return

    logger.info({
        "event": "image_parallel_generation_start",
        "n": request.n,
        "model": request.model,
    })
    # 每张图片一个线程，同时启动
    futures = {}
    results: dict[int, list[ImageOutput]] = {}
    errors: dict[int, Exception] = {}
    with ThreadPoolExecutor(max_workers=request.n) as executor:
        for index in range(1, request.n + 1):
            future = executor.submit(_generate_single_image, request, index, request.n)
            futures[future] = index

        # 按完成顺序收集结果
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                errors[index] = exc
                logger.warning({
                    "event": "image_parallel_generation_error",
                    "index": index,
                    "error": str(exc)[:300],
                })

    # yield 结果：跳过索引顺序限制，不再让低索引失败阻塞高索引成功结果
    emitted = False
    last_error = ""
    # 先 yield 所有成功的结果
    for index in range(1, request.n + 1):
        if index in results:
            for output in results[index]:
                emitted = True
                yield output
        elif index in errors:
            last_error = str(errors[index])
            if not emitted:
                logger.warning({
                    "event": "image_parallel_failure_before_success",
                    "failed_index": index,
                    "error": last_error[:200],
                })

    # 如果有失败但也有成功，记录警告
    if emitted:
        for index in range(1, request.n + 1):
            if index in errors:
                logger.warning({
                    "event": "image_parallel_partial_failure",
                    "failed_index": index,
                    "error": str(errors[index])[:200],
                })

    if not emitted:
        if not last_error:
            last_error = "no account in the pool could generate images — check account quota and rate-limit status"
        raise ImageGenerationError(image_stream_error_message(last_error), conversation_id="")


def stream_image_chunks(outputs: Iterable[ImageOutput]) -> Iterator[dict[str, Any]]:
    for output in outputs:
        yield output.to_chunk()


def collect_image_outputs(outputs: Iterable[ImageOutput]) -> dict[str, Any]:
    created = None
    data: list[dict[str, Any]] = []
    message = ""
    progress_parts: list[str] = []
    account_email = ""
    for output in outputs:
        created = created or output.created
        if output.account_email and not account_email:
            account_email = output.account_email
        if output.kind == "progress" and output.text:
            progress_parts.append(output.text)
        elif output.kind == "message":
            message = output.text
        elif output.kind == "result":
            data.extend(output.data)

    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if not data:
        text = message or "".join(progress_parts).strip()
        if text:
            result["message"] = text
    if account_email:
        result["_account_email"] = account_email
    return result
