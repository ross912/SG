"""DeepSeek Chat Completions 客户端，统一处理超时、错误与流式输出。"""
import json
import os
from collections.abc import Iterator
from typing import Any

import requests


DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekError(RuntimeError):
    """可安全展示给平台用户的 DeepSeek 调用错误。"""


def fast_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-v4-flash"


def analysis_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-v4-flash"


def is_configured() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    thinking: bool = False,
    json_output: bool = False,
    max_tokens: int = 1800,
    temperature: float = 0.2,
    timeout: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """执行一次非流式对话，返回正文和不含密钥的元数据。"""
    payload = _payload(
        messages,
        model=model,
        thinking=thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    if json_output:
        payload["response_format"] = {"type": "json_object"}

    response = _request(payload, stream=False, timeout=timeout)
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise DeepSeekError("DeepSeek 返回格式异常") from error
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekError("DeepSeek 返回了空内容")
    return content.strip(), {
        "model": body.get("model", payload["model"]),
        "usage": body.get("usage") or {},
    }


def stream_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 1800,
    temperature: float = 0.3,
    timeout: float | None = None,
) -> Iterator[str]:
    """把 DeepSeek SSE 响应转换为纯文本片段。"""
    payload = _payload(
        messages,
        model=model,
        thinking=False,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    response = _request(payload, stream=True, timeout=timeout)
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            line = (raw_line or "").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            if content:
                yield str(content)
    finally:
        response.close()


def _payload(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    thinking: bool,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict[str, Any]:
    selected_model = (model or fast_model()).strip()
    if not selected_model:
        raise DeepSeekError("DeepSeek 模型未配置")
    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "max_tokens": max(1, int(max_tokens)),
        "temperature": min(max(float(temperature), 0.0), 2.0),
        "stream": stream,
    }
    # 旧版 deepseek-chat 兼容接口不依赖 thinking 参数；仅 v4 显式传递。
    if selected_model.startswith("deepseek-v4"):
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if thinking and selected_model.startswith("deepseek-v4"):
        payload["reasoning_effort"] = "high"
    return payload


def _request(payload: dict[str, Any], *, stream: bool, timeout: float | None):
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise DeepSeekError("DeepSeek API 未配置")
    api_url = os.getenv("DEEPSEEK_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
    request_timeout = timeout or float(os.getenv("DEEPSEEK_TIMEOUT", "120"))
    try:
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=request_timeout,
            stream=stream,
        )
    except requests.RequestException as error:
        raise DeepSeekError(f"DeepSeek 网络请求失败：{type(error).__name__}") from error
    if response.ok:
        return response
    try:
        message = response.json().get("error", {}).get("message")
    except (ValueError, AttributeError):
        message = None
    response.close()
    detail = str(message or f"HTTP {response.status_code}")[:180]
    raise DeepSeekError(f"DeepSeek 调用失败：{detail}")
