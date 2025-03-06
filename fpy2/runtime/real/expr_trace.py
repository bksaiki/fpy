from ...ir import *
from titanfp.arithmetic.evalctx import EvalCtx

class ExprTrace:
    # TODO: add code location
    def __init__(self, expr: Exp, value: float | bool, env: dict[NamedId, float | bool], ctx: EvalCtx):
        self.expr = expr
        self.value = value
        self.env = env
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"ExprTrace(expr={self.expr}, value={self.value}, env={self.env}, ctx={self.ctx})"