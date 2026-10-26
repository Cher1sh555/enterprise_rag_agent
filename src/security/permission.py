"""权限过滤唯一入口：检索前过滤 + 检索后校验。

两条规则，简单到能背：

1. **密级是硬门槛**：`chunk.sensitivity` 的等级 ≤ 用户 `level` 的等级，否则不可读；
2. **部门是范围**：
   - 最低密级（`security_levels[0]`，如"普通"）的文档视为公司公开制度，**全员可读**；
   - 更高密级（如"机密"）只允许同部门，或挂在"公共"部门下。

未知密级 / 未知部门一律**不可读**（fail-closed），并打一条 warning。

为什么"普通级全员可读"：部门标签表示**数据归属**，不是**访问边界**。
HR 的年假政策如果只有 HR 能查，知识库就失去意义了；真正的隔离靠密级 + 机密文档的部门范围。

为什么不只做后置过滤：后置过滤时越权片段已经进了候选集和 prompt，
模型可能直接说出来；前置过滤让它在**进入候选集之前**就不存在。
"""

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
