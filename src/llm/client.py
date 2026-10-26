"""LLM 调用封装：超时 + 1 次指数退避重试 + 异常映射。

v0 不做：缓存、限流、降级、流式（M4）。
提示词先写在常量里；v1 挪到 configs/prompts.yaml。
"""

from __future__ import annotations

import os

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.schema import Chunk
from src.utils.exceptions import ConfigError, LLMCallError, LLMRateLimitError, LLMTimeoutError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是企业内部知识库问答助手。
规则：
1. 只能依据 <doc> 标签内的内容回答，不得使用外部知识。
2. 每条结论后用 [编号] 标注依据，编号对应 <doc id="N">。
3. 如果 <doc> 中没有足够依据，直接回答"知识库中未找到相关依据"，不要猜测。"""


def chat(messages: list[dict], *, timeout: float = 30.0, model: str | None = None) -> str:
    client = _client()
    try:
        response = _request(client, messages, timeout, model)
    except (APITimeoutError, APIConnectionError) as exc:
        raise LLMTimeoutError("LLM 调用超时或连接失败", {"model": model}) from exc
    except RateLimitError as exc:
        raise LLMRateLimitError("LLM 触发限流", {"model": model}) from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMCallError("LLM 调用失败", {"error": repr(exc)}) from exc
    return (response.choices[0].message.content or "").strip()


@retry(
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
    wait=wait_exponential(multiplier=0.5, max=4),
    stop=stop_after_attempt(2),
    reraise=True,
)
def _request(client: OpenAI, messages, timeout: float, model: str | None):
    return client.chat.completions.create(
        model=model or os.getenv("LLM_MODEL") or "deepseek-chat",
        messages=messages,
        timeout=timeout,
        temperature=0.2,
    )


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    docs = "\n\n".join(
        f'<doc id="{i}">\n来源: {c.source} 第 {c.page_no} 页\n{c.text}\n</doc>'
        for i, c in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{docs}\n\n问题：{question}"},
    ]


def _client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("缺少 LLM_API_KEY", {"env": "LLM_API_KEY", "hint": "填 .env 或设环境变量"})
    return OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or "https://api.deepseek.com")