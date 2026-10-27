"""LLM 调用封装：限流 -> 缓存 -> 超时重试 -> 降级。

调用顺序是固定的，别调换：
1. 限流（保护自己也保护上游配额）
2. 查缓存（省一次调用）
3. 调模型（超时 + 指数退避重试；只重试超时/连接/限流，4xx 不重试）
4. 写缓存

降级：主模型限流/超时且配置了 fallback → 换模型重试，返回 `degraded=True`。
**不做静默降级** —— 调用方要把 degraded 传下去（审计 / UI 提示）。

模型解析顺序：显式参数 > 环境变量 LLM_MODEL > settings.yaml 的 models.llm_model。
（此前 yaml 里的 llm_model 是死配置，只读环境变量；这里统一了。）
"""

from __future__ import annotations

import os

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.llm.cache import ResponseCache, make_key
from src.llm.ratelimit import TokenBucket
from src.schema import Chunk
from src.utils.exceptions import (
    ConfigError,
    LLMCallError,
    LLMRateLimitError,
    LLMTimeoutError,
    RAGError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TEMPERATURE = 0.2
DEFAULT_BASE_URL = "https://api.deepseek.com"

_SYSTEM_PROMPT = """你是企业内部知识库问答助手。
规则：
1. 只能依据 <doc> 标签内的内容回答，不得使用外部知识。
2. 每条结论后用 [编号] 标注依据，编号对应 <doc id="N">。
3. 如果 <doc> 中没有足够依据，直接回答"知识库中未找到相关依据"，不要猜测。"""

# 进程级单例：缓存与令牌桶都应该是全局的
_cache = ResponseCache()
_bucket = TokenBucket(
    rate=float(os.getenv("LLM_RATE_RPS", "5")),
    capacity=int(os.getenv("LLM_RATE_BURST", "10")),
)


def resolve_model(model: str | None = None) -> str:
    if model:
        return model
    env = os.getenv("LLM_MODEL", "").strip()
    return env or get_settings().models.llm_model


def resolve_fallback_model() -> str | None:
    env = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    return env or get_settings().models.fallback_model


def chat(
    messages: list[dict],
    *,
    timeout: float = 30.0,
    model: str | None = None,
    use_cache: bool = True,
) -> str:
    chosen = resolve_model(model)
    _bucket.acquire()

    key = make_key(chosen, messages, TEMPERATURE)
    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            logger.debug("LLM 缓存命中: model=%s", chosen)
            return cached

    text = _call(chosen, messages, timeout)
    if use_cache:
        _cache.put(key, text)
    return text


def chat_with_fallback(
    messages: list[dict],
    *,
    primary: str | None = None,
    fallback: str | None = None,
    timeout: float = 30.0,
) -> tuple[str, bool]:
    """返回 (answer, degraded)。降级只在限流/超时时发生。"""
    try:
        return chat(messages, model=primary, timeout=timeout), False
    except (LLMRateLimitError, LLMTimeoutError) as exc:
        target = fallback or resolve_fallback_model()
        if not target:
            raise
        logger.warning("主模型失败，降级到 %s：%s", target, exc)
        return chat(messages, model=target, timeout=timeout, use_cache=False), True


def build_messages(
    question: str,
    chunks: list[Chunk],
    history: list[dict] | None = None,
) -> list[dict]:
    """拼接 prompt。history 是最近几轮 [{'role','content'}]，放在 system 之后。"""
    docs = "\n\n".join(
        f'<doc id="{i}">\n来源: {c.source} 第 {c.page_no} 页\n{c.text}\n</doc>'
        for i, c in enumerate(chunks, start=1)
    )
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": f"{docs}\n\n问题：{question}"})
    return messages


def cache_stats() -> dict:
    return {
        "size": len(_cache),
        "hits": _cache.hits,
        "misses": _cache.misses,
        "hit_rate": round(_cache.hit_rate, 3),
    }


def reset_llm_state() -> None:
    """测试用：清空缓存（令牌桶保留，避免影响限流语义）。"""
    _cache.clear()


def _call(model: str, messages: list[dict], timeout: float) -> str:
    client = _client()
    try:
        response = _request(client, messages, timeout, model)
    except (APITimeoutError, APIConnectionError) as exc:
        raise LLMTimeoutError("LLM 调用超时或连接失败", {"model": model}) from exc
    except RateLimitError as exc:
        raise LLMRateLimitError("LLM 触发限流", {"model": model}) from exc
    except RAGError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMCallError("LLM 调用失败", {"error": repr(exc)}) from exc
    return (response.choices[0].message.content or "").strip()


@retry(
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
    wait=wait_exponential(multiplier=0.5, max=4),
    stop=stop_after_attempt(2),
    reraise=True,
)
def _request(client: OpenAI, messages, timeout: float, model: str):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
        temperature=TEMPERATURE,
    )


def _client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "缺少 LLM_API_KEY", {"env": "LLM_API_KEY", "hint": "填 .env 或设环境变量"}
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL,
    )
