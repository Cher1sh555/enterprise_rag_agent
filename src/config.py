"""配置加载与校验：读取 configs/settings.yaml 与 .env。"""
from __future__ import annotations  #让所有类型注解 "延迟求值

from pathlib import Path
from functools import lru_cache  #把函数的返回值按参数缓存起来 —— 同一份输入再来时，直接取缓存，不再真跑函数
from pydantic import BaseModel,Field,ValidationError,model_validator

from utils.exceptions import RetrievalError


#BaseModel数据类基模定义 State、Tool 参数、LLM 返回结构
# Field字段级约束设默认值、描述、大小 / 长度 / 正则、别名
# ValidationError校验异常捕获后把错误回喂给 LLM 重试（ReAct 关键一环）
# model_validator跨字段校验器当字段间有依赖（起止时间、互斥开关）时整模型校验

class AppConfig(BaseModel): ...
class PathConfig(BaseModel):
    raw_docs: Path; processed: Path; index: Path; evaluation: Path;

class ModelsConfig(BaseModel):
   llm_api:str; llm_model:str;
   embedding_model:str;
   rerank_model:str;

class ChunkingConfig(BaseModel):
    chunk_size:int = Field(gt=54)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size，否则切分会原地打转")
        return self

class RetrievalConfig(BaseModel):
    top_k : int; rerank_topo_k : int; relevance_threshold : float;
    @model_validator(mode="after")
    def _rerank_le_topk(self) -> RetrievalConfig:
        if self.rerank_top_k > self.top_k:
            raise ValueError("rerank_top_k 必须小于等于 top_k")
        return self

class settings(BaseModel):
    app: AppConfig; paths: PathConfig; models: ModelsConfig;
    chunking: ChunkingConfig; retrieval: RetrievalConfig;
    security: SecurityConfig; audit:AuditConfig; ingest: IngestConfig

    @property
    def default_sensitivity(self) -> str:
        #none: 未设定密级时，默认按敏感级别最低的处理
        ...

def load_settings(path:Path | None = None) -> settings:
    """读 configs/settings.yaml + .env，交给pydantic校验后返回全新setting对象。失败 -> ConfigError(detail=...)。"""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载load_settings返回的settings 不再读取文件
    进程内单例。测试用 get_settings.cache_clear() 复位。
    get_settings.cache_clear()清除缓存后 下次重新load_settings()"""


