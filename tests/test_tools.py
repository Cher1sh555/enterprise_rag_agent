"""工具契约：算得对、拒绝注入、错误不抛。"""

from __future__ import annotations

import pytest

from src.agent.tools import TOOLS, calculator, run_tool


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1+2*3", "7"),
        ("(8-2)/3", "2"),
        ("-5+2", "-3"),
        ("10%3", "1"),
        ("(600*2+400)*1.1", "1760"),      # 浮点尾差要消掉
    ],
)
def test_calculator_arithmetic(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize(
    "payload",
    [
        '__import__("os")',      # 代码注入
        'open("x")',             # 文件访问
        '"a" * 3',               # 字符串运算
        "9**9**9",               # 指数炸弹
        "x + 1",                 # 未定义变量
    ],
)
def test_calculator_rejects_unsafe_input(payload):
    with pytest.raises(ValueError):
        calculator(payload)


def test_calculator_rejects_too_long():
    with pytest.raises(ValueError):
        calculator("1+" * 60 + "1")


def test_run_tool_returns_error_string_instead_of_raising():
    assert run_tool("calculator", "1/0").startswith("[工具错误]")
    assert run_tool("no_such_tool", "1") == "[工具错误] 未知工具: no_such_tool"


def test_tool_registry_has_description():
    spec = TOOLS["calculator"]

    assert spec.description and spec.func is calculator
