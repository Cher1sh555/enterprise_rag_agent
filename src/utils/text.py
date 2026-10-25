"""文本原语：分词与可见字符统计的唯一实现。

为什么单独成文件：
    tokenize 同时被"建索引（BM25 语料）"和"查询（BM25 打分）"使用。
    如果查询端从 indexer 导入，retrieval 就会反向依赖 ingest（方向错了）。
    放在 utils 谁都能用，并且保证索引端与查询端是同一套分词——
    这一处不一致会让 BM25 全量零分，而且不报错。
"""

from __future__ import annotations

import logging
import re

import jieba

jieba.setLogLevel(logging.WARNING)  # 关掉首次构建词典时的 DEBUG 刷屏

# 纯标点 / 空白 / 控制字符：分词后要剔除的"无信息 token"
_NOISE = re.compile(r"[\s\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """中文分词并剔除纯标点，返回顺序稳定的 token 列表。

    >>> tokenize("一线城市差旅住宿标准？")
    ['一线', '城市', '差旅', '住宿', '标准']
    """
    if not text:
        return []
    return [
        word
        for word in (w.strip() for w in jieba.lcut(text))
        if word and not _NOISE.fullmatch(word)
    ]


def visible_chars(text: str) -> int:
    """可见字符数（忽略空白与标点），统计与断言统一口径。"""
    if not text:
        return 0
    return len(_NOISE.sub("", text))


def char_bigrams(text: str) -> list[str]:
    """字符二元组（去空白），供 HashEmbedder 做相似度近似。"""
    stripped = "".join(ch for ch in text if not ch.isspace())
    if not stripped:
        return []
    return [stripped[i : i + 2] for i in range(len(stripped) - 1)] or [stripped]


if __name__ == "__main__":  # 手工自检：python -m src.utils.text
    sample = "第7条 一线城市差旅住宿标准为每晚不超过 600 元。"
    print("tokens :", tokenize(sample))
    print("visible:", visible_chars(sample))
    print("bigrams:", char_bigrams("差旅标准"))
