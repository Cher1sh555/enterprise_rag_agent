"""自定义工具（计算器等）。

安全求值原则：**完全不用 `eval`**，而是用 AST 白名单递归求值。
输入来自用户，`eval("__import__('os').system(...)")` 是真实攻击面；
白名单之外的一切节点（Call / Attribute / Subscript / Name / Lambda…）直接拒绝。

可扩展点：新增工具只需写函数 + 注册进 `TOOLS`，图里通过 `run_tool(name, arg)` 调用。
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass

MAX_EXPR_LEN = 100
MAX_POW_EXPONENT = 10

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[[str], str]


def calculator(expr: str) -> str:
    """四则运算。非法输入抛 ValueError（由 run_tool 转成错误串）。"""
    text = (expr or "").strip()
    if not text:
        raise ValueError("表达式为空")
    if len(text) > MAX_EXPR_LEN:
        raise ValueError("表达式过长")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"不是合法表达式: {text}") from exc

    value = _eval_node(tree.body)
    if isinstance(value, float):
        value = round(value, 10)        # 消掉 1760.0000000000002 这类浮点尾差
        if value.is_integer():
            value = int(value)
    return str(value)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字常量")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise ValueError("指数过大")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    raise ValueError(f"不允许的语法: {type(node).__name__}")


TOOLS: dict[str, ToolSpec] = {
    "calculator": ToolSpec(
        name="calculator",
        description="四则运算计算器，输入形如 (600*2+400)*1.1 的表达式",
        func=calculator,
    )
}


def run_tool(name: str, arg: str) -> str:
    """统一入口：未知工具/执行失败都返回错误串，绝不抛给图。"""
    spec = TOOLS.get(name)
    if spec is None:
        return f"[工具错误] 未知工具: {name}"
    try:
        return spec.func(arg)
    except Exception as exc:  # noqa: BLE001 - 工具错误不该打断整条链路
        return f"[工具错误] {exc}"
