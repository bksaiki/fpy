"""
MPFX: common utilities
"""

from dataclasses import dataclass

from ...ast import FuncDef
from ..backend import CompileError

__all__ = [
    'MPFXCompileError',
    'CompileCtx',
    'CppOptions',
]

class MPFXCompileError(CompileError):
    """Compiler error for MPFX backend"""

    def __init__(self, func: FuncDef, msg: str, *args):
        lines: list[str] = [f'MPFX backend: {msg} in function `{func.name}`']
        lines.extend(str(arg) for arg in args)
        super().__init__('\n '.join(lines))

@dataclass
class CompileCtx:
    ctx_name: str | None
    lines: list[str]
    indent_str: str
    indent_level: int

    @staticmethod
    def default(ctx_name: str, indent_str: str = ' ' * 4):
        return CompileCtx(ctx_name, [], indent_str, 0)

    def indent(self):
        return CompileCtx(self.ctx_name, self.lines, self.indent_str, self.indent_level + 1)

    def dedent(self):
        assert self.indent_level > 0
        return CompileCtx(self.ctx_name, self.lines, self.indent_str, self.indent_level - 1)

    def with_ctx(self, ctx_name: str | None):
        return CompileCtx(ctx_name, self.lines, self.indent_str, self.indent_level)

    def add_line(self, line: str):
        self.lines.append(self.indent_str * self.indent_level + line)

@dataclass(frozen=True)
class CppOptions:
    unsafe_finitize_int: bool
    unsafe_cast_int: bool
