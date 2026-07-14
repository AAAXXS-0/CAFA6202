"""FinixDoc-VL API 隔离层。

赛事官方接口使用 multipart/form-data 上传图片，响应外层和模型结果层各有一层
JSON。本模块同时保留 Chat Completions 协议，便于本地兼容服务或后续接口切换；
切图、缓存和 Markdown 聚合代码不需要感知具体协议。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any

import requests


OFFICIAL_PROTOCOL = "official_multipart"
CHAT_PROTOCOL = "chat_completions"


class FinixDocTemporaryError(RuntimeError):
    """服务繁忙、限流等适合稍后重试的临时错误。"""


class FinixDocPermanentError(RuntimeError):
    """凭据、字段或业务响应错误，原样重试通常不会恢复。"""


def _strip_markdown_fence(text: str) -> str:
    """只移除包住整个响应的 Markdown 代码围栏，不改动正文内部代码块。"""

    value = text.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*\n?(.*?)\n?```", value, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _chat_content(payload: dict[str, Any]) -> str:
    """从 OpenAI 风格 choices 中提取文本，兼容字符串和文本块数组。"""

    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = _strip_markdown_fence(str(content))
    if not text:
        raise RuntimeError("FinixDoc-VL 返回了空内容")
    return text


def parse_official_response(payload: dict[str, Any]) -> str:
    """解析赛事接口的“外层 result → 内层 result → choices”响应。"""

    if payload.get("success") is not True:
        message = payload.get("message") or "接口返回 success=false"
        raise FinixDocPermanentError(f"FinixDoc-VL 业务请求失败：{message}")

    outer_result = payload.get("result")
    if not isinstance(outer_result, dict):
        raise FinixDocPermanentError("FinixDoc-VL 响应缺少 result 对象")
    nested = outer_result.get("result")
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except json.JSONDecodeError as error:
            raise FinixDocPermanentError("FinixDoc-VL 内层 result 不是有效 JSON") from error
    if not isinstance(nested, dict):
        raise FinixDocPermanentError("FinixDoc-VL 响应缺少内层模型结果")
    try:
        return _chat_content(nested)
    except (KeyError, IndexError, TypeError) as error:
        raise FinixDocPermanentError("FinixDoc-VL 内层响应缺少 choices.message.content") from error


class FinixDocClient:
    """按指定协议调用 FinixDoc-VL，并统一返回无外层围栏的 Markdown。"""

    def __init__(
        self,
        api_url: str,
        model: str = "FinixDoc-VL",
        api_key_env: str = "FINIXDOC_API_KEY",
        timeout: int = 240,
        max_retries: int = 4,
        *,
        protocol: str = OFFICIAL_PROTOCOL,
        user_id: str | None = None,
        user_id_env: str = "FINIXDOC_USER_ID",
        api_key: str | None = None,
    ) -> None:
        if not api_url:
            raise ValueError("api_url 不能为空")
        if protocol not in {OFFICIAL_PROTOCOL, CHAT_PROTOCOL}:
            raise ValueError(f"不支持的 FinixDoc-VL 协议：{protocol}")
        self.api_url = api_url
        self.api_model = model
        self.model = f"{model}@{protocol}"
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self.protocol = protocol
        self.user_id = user_id
        self.user_id_env = user_id_env
        self._api_key = api_key

    @classmethod
    def from_official_doc(
        cls,
        path: str | Path,
        *,
        user_id: str | None = None,
        model: str = "FinixDoc-VL",
        timeout: int = 240,
        max_retries: int = 4,
    ) -> "FinixDocClient":
        """从官方调用说明读取地址和固定凭据，凭据不会写入代码或清单。"""

        text = Path(path).read_text(encoding="utf-8-sig")
        url_match = re.search(r"https://[^\s'\"]+/api/finix_doc/call_with_file", text)
        key_match = re.search(r"apiKey=([A-Za-z0-9]+)", text)
        users = re.findall(r"(?m)^finix[A-Z]\d{4}$", text)
        if not url_match or not key_match or not users:
            raise ValueError("无法从官方调用说明解析 API 地址、apiKey 或 userId")
        selected_user = user_id or users[0]
        if selected_user not in users:
            raise ValueError(f"userId 不在官方白名单中：{selected_user}")
        return cls(
            api_url=url_match.group(0),
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            protocol=OFFICIAL_PROTOCOL,
            user_id=selected_user,
            api_key=key_match.group(1),
        )

    def _credentials(self) -> tuple[str, str]:
        api_key = self._api_key or os.environ.get(self.api_key_env)
        user_id = self.user_id or os.environ.get(self.user_id_env)
        if not api_key:
            raise RuntimeError(f"未设置 API Key：请设置 {self.api_key_env} 或使用官方说明文件")
        if not user_id:
            raise RuntimeError(f"未设置 userId：请设置 {self.user_id_env} 或传入 --user-id")
        return api_key, user_id

    def _chat_payload(self, image_path: Path, prompt: str) -> dict[str, Any]:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "model": self.api_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        }

    def _recognize_once(self, image_path: Path, prompt: str) -> str:
        if self.protocol == CHAT_PROTOCOL:
            headers: dict[str, str] = {}
            api_key = self._api_key or os.environ.get(self.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                self.api_url,
                json=self._chat_payload(image_path, prompt),
                headers=headers,
                timeout=(15, self.timeout),
            )
            response.raise_for_status()
            return _chat_content(response.json())

        api_key, user_id = self._credentials()
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        with image_path.open("rb") as image_file:
            response = requests.post(
                self.api_url,
                data={
                    "userId": user_id,
                    "apiKey": api_key,
                    "fileName": image_path.name,
                },
                files={"file": (image_path.name, image_file, mime)},
                timeout=(15, self.timeout),
            )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "json" not in content_type:
            preview = re.sub(r"\s+", " ", response.text)[:240]
            if "服务器繁忙" in response.text or "顾客太多" in response.text:
                raise FinixDocTemporaryError(
                    "官方接口返回服务器繁忙页面（HTTP 200、非 JSON）"
                )
            raise FinixDocPermanentError(
                f"官方接口返回非 JSON 内容：content-type={content_type!r}，{preview}"
            )
        try:
            payload = response.json()
        except requests.JSONDecodeError as error:
            raise FinixDocTemporaryError("官方接口返回了无法解析的 JSON") from error
        return parse_official_response(payload)

    def recognize(self, image_path: Path, prompt: str) -> str:
        """识别一张图片；可重试网络错误、限流和服务端临时错误。"""

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._recognize_once(Path(image_path), prompt)
            except FinixDocPermanentError as error:
                last_error = error
                break
            except (requests.RequestException, FinixDocTemporaryError) as error:
                last_error = error
                # 参数错误和普通 4xx 重试无意义；429 限流仍按退避策略重试。
                if isinstance(error, requests.HTTPError) and error.response is not None:
                    status = error.response.status_code
                    if 400 <= status < 500 and status != 429:
                        break
                if attempt + 1 < self.max_retries:
                    delay = min(2**attempt, 8)
                    print(
                        f"[FinixDoc-VL 重试 {attempt + 2}/{self.max_retries}] "
                        f"{Path(image_path).name}：{error}；{delay} 秒后重试",
                        flush=True,
                    )
                    time.sleep(delay)
        raise RuntimeError(f"FinixDoc-VL 请求失败：{last_error}") from last_error
