"""
Triton backend: the normal form.

The cpp backend's ``StatementForm``, with calls inlined away and one exit.
An ``if`` statement is kept.  A kernel has no per-lane branch, so the emitter
flattens it under a mask -- but after the analyses, which read its guard:
if-converting here would run each arm on every input *in the program*, and
format inference would rightly bound it over all of them.

A comprehension becomes a loop, which the emitter runs across a tile's lanes.

What survives normalization and should not is an error, per the contract's
rejection principle -- a refusal is acceptable, a different answer is not.
"""

from ...analysis import DefineUse, Reachability
from ...ast import (
    Call,
    DefaultVisitor,
    Expr,
    FuncDef,
    NamedId,
    Var,
    WhileStmt,
)
from ...function import Function
from ...module import Module
from ...transform import (
    FuncInline,
    RescaleFixed,
    SingleExit,
    StatementForm,
    TransformDeclined,
)
from ..backend import CompileError

__all__ = ['TritonNormalizeError', 'normalize', 'normalize_module']


class TritonNormalizeError(CompileError):
    """A program that does not reach the Triton normal form."""


_MAX_ROUNDS = 4
"""Cap on the rounds below.

One productive application suffices across the corpus -- 67 of 86 functions
are already normal and the other 19 are stable after one -- so this only
bounds a pipeline that stops converging, rather than a case that needs it.
"""


class _NotNormal(DefaultVisitor):
    """What remains that the normal form does not admit."""

    def __init__(self, func: FuncDef):
        super().__init__()
        self.func = func
        self.def_use = DefineUse.analyze(func)
        self.reasons: list[str] = []

    def _visit_statement(self, stmt, ctx):
        match stmt:
            case WhileStmt():
                # a tile-wide loop must run a fixed number of times; a
                # condition the body moves makes the count per-lane
                moved = self.def_use.mutated_in(stmt.body)
                if any(v in moved for v in _reads(stmt.cond)):
                    self.reasons.append('a `while` condition varies')
        return super()._visit_statement(stmt, ctx)

    def _visit_call(self, e: Call, ctx):
        if isinstance(e.fn, Function):
            self.reasons.append(f'a call to `{e.fn.name}` remains')
        return super()._visit_call(e, ctx)

    def check(self) -> list[str]:
        self._visit_function(self.func, None)
        n = len(Reachability.analyze(self.func).ret_stmts)
        if n > 1:
            self.reasons.append(f'{n} `return`s remain')
        return self.reasons


class _Reads(DefaultVisitor):
    """Names an expression reads."""

    def __init__(self):
        super().__init__()
        self.names: set[NamedId] = set()

    def _visit_var(self, v: Var, ctx):
        self.names.add(v.name)


def _reads(e: Expr) -> set[NamedId]:
    v = _Reads()
    v._visit_expr(e, None)
    return v.names


def normalize(func: FuncDef) -> FuncDef:
    """*func* in the Triton normal form, assuming its callees already are.

    ``SingleExit`` runs **first**, not last: ``FuncInline`` refuses a callee
    with more than one return, so a function must be single-exit before anyone
    can inline it.  That makes the order across a call graph leaves-first,
    which is what :func:`normalize_module` is for -- this function alone cannot
    fix a callee, since it only holds the caller.

    ``StatementForm`` runs **either side of the inline**: `FuncInline`
    splices a callee's body into the enclosing *statement* list, so it cannot
    take a call inside a comprehension until that is a loop, and a callee
    brings comprehensions of its own.

    Raises :class:`TritonNormalizeError` if the form is not reached; the
    passes' own :class:`~fpy2.transform.TransformDeclined` propagates as-is,
    since it already says which construct refused.
    """
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")

    for _ in range(_MAX_ROUNDS):
        func = SingleExit.apply(func)
        func = StatementForm.apply(func)
        func = FuncInline.apply(func, recursive=True)
        func = StatementForm.apply(func)
        # after the statement form: it emits the scale-in and scale-out as
        # statements, which a rounding inside a comprehension has no slot for
        try:
            func = RescaleFixed.apply(func)
        except TransformDeclined:
            pass  # nothing to move is not a failure
        reasons = _NotNormal(func).check()
        if not reasons:
            return func
    # the loop is meant to converge -- inlining is bounded by an acyclic call
    # graph, and lowering creates no work for it.  Reaching the bound means
    # one of those is false.
    raise TritonNormalizeError(
        f'`{func.name}` did not converge in {_MAX_ROUNDS} rounds, which is a '
        f'bug in the normal form rather than a program it declines; what '
        f'remains: ' + '; '.join(sorted(set(reasons)))
    )


def normalize_module(module: Module) -> Module:
    """Every function in *module* in the Triton normal form.

    :meth:`Module.map` walks leaves-first and rebinds each caller's ``Call.fn``
    to the already-transformed callee, which is what makes the order work: a
    callee is single-exit by the time its caller reaches :func:`normalize` and
    tries to inline it.
    """
    if not isinstance(module, Module):
        raise TypeError(f"Expected a 'Module', got {module}")
    return module.map(lambda _m, fd: normalize(fd))
