"""配置加载与校验：读取 configs/settings.yaml 与 .env。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator

from src.utils.exceptions import ConfigError
from src.utils.logger import project_root


class AppConfig(BaseModel):
    name: str
    debug: bool = False


class PathConfig(BaseModel):
    raw_docs: Path
    processed: Path
    index: Path
    evaluation: Path

    @model_validator(mode="after")
    def _absolutize(self) -> PathConfig:
        """相对路径一律相对项目根解析，不随 cwd 漂移。"""
        for name in ("raw_docs", "processed", "index", "evaluation"):
            value = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, project_root() / value)
        return self

class ModelsConfig(BaseModel):
    llm_api: str
    llm_model: str
    embedding_model: str
    rerank_model: str
    fallback_model: str | None = None   # 主模型限流/超时时降级用；None = 不降级


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(ge=64)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size，否则切分会原地打转")
        return self


class RetrievalConfig(BaseModel):
    top_k: int
    rerank_top_k: int
    relevance_threshold: float

    @model_validator(mode="after")
    def _rerank_le_topk(self) -> RetrievalConfig:
        if self.rerank_top_k > self.top_k:
            raise ValueError("rerank_top_k 必须小于等于 top_k")
        return self


class SecurityConfig(BaseModel):
    departments: list[str]
    security_levels: list[str]


class AuditConfig(BaseModel):
    log_dir: Path
    enabled: bool = True


class IngestConfig(BaseModel):
    scan_threshold_chars_per_page: int = 50
    render_dpi: int = 200
    ocr_lang: str = "ch"
    ocr_use_gpu: bool = False
    ocr_min_confidence: float = 0.5
    max_pages: int | None = None
    unknown_department: str = "未知"
    default_sensitivity: str | None = None
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md"]
    )


class Settings(BaseModel):
    app: AppConfig
    paths: PathConfig
    models: ModelsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    security: SecurityConfig
    audit: AuditConfig
    ingest: IngestConfig

    @property
    def default_sensitivity(self) -> str:
        """未显式配置时取最严格档（fail closed）。

        反例：默认成最低档会把没打标签的机密文档当普通文档放行，
        属于静默越权，所以这里取 security_levels 的最后一档。
        """
        return self.ingest.default_sensitivity or self.security.security_levels[-1]


def load_settings(path: Path | None = None) -> Settings:
    """读 configs/settings.yaml + .env，交给 pydantic 校验后返回 Settings。失败 -> ConfigError。"""
    load_dotenv()  # 必须在读取任何环境变量之前调用

    config_path = Path(path) if path is not None else project_root() / "configs" / "settings.yaml"
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    if not config_path.is_file():
        raise ConfigError("配置文件不存在", {"path": str(config_path)})

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError("settings.yaml 解析失败", {"path": str(config_path), "error": str(exc)}) from exc

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            "配置校验失败", {"path": str(config_path), "errors": exc.errors()}
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试用 get_settings.cache_clear() 复位。"""
    return load_settings()
