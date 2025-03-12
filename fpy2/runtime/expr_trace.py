from dataclasses import dataclass
from titanfp.arithmetic.evalctx import EvalCtx
from titanfp.arithmetic.ieee754 import Float

from ..ir import *

@dataclass
class ExprTraceEntry:
    expr: Expr
    value: bool | Float
    env: dict[NamedId, bool | Float]
    ctx: EvalCtx
