"""部门 / 密级标签规范与校验。"""
from __future__ import annotations
from dataclasses import dataclass
"""自动生成类
@dataclass(init=True, repr=True, eq=True, frozen=False)
- `init=True`：自动生成 `__init__`
- `repr=True`：自动生成好看的打印
- `eq=True`：自动按字段值比较相等
- `frozen=True`：冻结对象，实例创建之后不能修改属性（只读）"""
from pathlib import Path