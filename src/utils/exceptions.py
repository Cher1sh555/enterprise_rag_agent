"""统一异常体系。

设计原则：
1. 所有可预期异常都继承 RAGError —— 上层只捕获这一个基类就能兜住全部业务错误，
   不会把 KeyError / FileNotFoundError 这类"意外"和"业务失败"混在一起。
2. 异常自带 message + detail，detail 是结构化字段（路径、doc_id、用户身份等），
   方便日志和审计直接落库，不用再拼字符串。
3. 按"阶段"分组，和服务模块一一对应：谁抛的错，一眼能看出从哪来。
"""

from __future__ import annotations

from typing import Any, Mapping


class RAGError(Exception):
    """项目内所有可预期异常的基类。"""

    def __init__(self, message: str, detail: Mapping[str, Any] | None = None) -> None:
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if not self.detail:
            return self.message
        kv = ", ".join(f"{k}={v!r}" for k, v in self.detail.items())
        return f"{self.message} ({kv})"


# ------------------------------------------------------------------
# 配置 / 启动阶段（阶段 2）
# ------------------------------------------------------------------
class ConfigError(RAGError):
    """配置缺失或非法：settings.yaml 字段错误、环境变量未设置等。"""


# ------------------------------------------------------------------
# 离线索引管线（阶段 3）
# ------------------------------------------------------------------
class LoadError(RAGError):
    """文档读取失败：文件不存在、损坏、加密、编码异常。"""


class UnsupportedFormatError(LoadError):
    """不支持的文档格式（后缀不在白名单内）。"""


class CleanError(RAGError):
    """清洗阶段失败。"""


class SplitError(RAGError):
    """分块阶段失败：配置非法、切出空块等。"""


class EmbeddingError(RAGError):
    """向量化失败：模型加载失败、显存/内存不足、输出维度异常。"""


class IndexBuildError(RAGError):
    """索引构建或落盘失败。"""


class IndexNotFoundError(RAGError):
    """索引文件缺失 —— 提示用户先跑 scripts/build_index.py。"""


# ------------------------------------------------------------------
# 在线检索（阶段 4、5）
# ------------------------------------------------------------------
class RetrievalError(RAGError):
    """检索阶段失败：索引损坏、查询向量维度不匹配等。"""


class PermissionDeniedError(RAGError):
    """权限校验未通过。

    注意：项目里永远不要用内置的 PermissionError 命名，
    否则会遮蔽内置异常，导致 except 捕获范围出现难以察觉的偏差。
    """


# ------------------------------------------------------------------
# 大模型调用（阶段 7）
# ------------------------------------------------------------------
class LLMCallError(RAGError):
    """LLM 调用失败（基类）。"""


class LLMTimeoutError(LLMCallError):
    """调用超时 —— 通常可重试。"""


class LLMRateLimitError(LLMCallError):
    """触发限流（429）—— 需要退避后重试。"""


# ------------------------------------------------------------------
# 安全与审计（阶段 8）
# ------------------------------------------------------------------
class AuditError(RAGError):
    """审计日志写入失败。

    约定：审计失败只上报告警，不允许中断主问答链路。
    """

class OCRError(LoadError):
    """OCR 引擎不可用或识别失败。"""

__all__ = [
    "RAGError",
    "ConfigError",
    "LoadError",
    "UnsupportedFormatError",
    "CleanError",
    "SplitError",
    "EmbeddingError",
    "IndexBuildError",
    "IndexNotFoundError",
    "RetrievalError",
    "PermissionDeniedError",
    "LLMCallError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "AuditError",
]
