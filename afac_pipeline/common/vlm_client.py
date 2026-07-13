"""FinixDoc-VL API 隔离层。

赛事 API 说明当前不在仓库中，所以这里按常见的 OpenAI Chat Completions
图片消息格式实现。若主办方字段不同，只需替换本文件，不必改动切图、缓存和
聚合代码。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FinixDocClient:
    def __init__(
        self,
        api_url: str,
        model: str = "FinixDoc-VL",
        api_key_env: str = "FINIXDOC_API_KEY",
        timeout: int = 180,
        max_retries: int = 4,
    ) -> None:
        if not api_url:
            raise ValueError("api_url 不能为空")
        self.api_url = api_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries

    def _payload(self, image_path: Path, prompt: str) -> bytes:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = {
            "model": self.model,
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
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def recognize(self, image_path: Path, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = Request(self.api_url, data=self._payload(image_path, prompt), headers=headers)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                text = str(content).strip()
                if not text:
                    raise RuntimeError("FinixDoc-VL 返回了空内容")
                return text.removeprefix("```markdown").removesuffix("```").strip()
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError, RuntimeError) as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"FinixDoc-VL 请求失败：{last_error}") from last_error
