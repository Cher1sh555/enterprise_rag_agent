from __future__ import annotations

from collections.abc import Sequence

from src.config import Settings
from src.schema import Chunk, User
from src.utils.logger import get_logger

logger = get_logger(__name__)

PUBLIC_DEPARTMENT = "公共"      # 任何部门的人都能读


def level_order(settings: Settings) -> dict[str, int]:
    """密级名 -> 等级（越大越严格）。"""
    return {level: index for index, level in enumerate(settings.security.security_levels)}


def can_read(user: User, chunk: Chunk, settings: Settings) -> bool:
    order = level_order(settings)
    user_level = order.get(user.level)
    chunk_level = order.get(chunk.sensitivity)
    if user_level is None or chunk_level is None:
        logger.warning(
            "未知密级，按不可读处理: user_level=%s chunk_level=%s",
            user.level,
            chunk.sensitivity,
        )
        return False

    if chunk_level > user_level:
        return False

    if chunk.sensitivity == settings.security.security_levels[0]:
        return True                              # 最低密级：全员可读

    return chunk.department in (user.department, PUBLIC_DEPARTMENT)


def allowed_ids(chunks: Sequence[Chunk], user: User, settings: Settings) -> set[str]:
    """检索前过滤用的 chunk_id 白名单。

    注意：返回**空集合**表示"一条都看不到"，调用方必须和 `None`（不过滤）区分开。
    """
    return {chunk.chunk_id for chunk in chunks if can_read(user, chunk, settings)}
