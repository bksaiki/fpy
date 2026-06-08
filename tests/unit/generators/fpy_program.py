"""
Type-directed Hypothesis generators for FPy programs.

Phase 2 scope:
- Two target types: ``RealType`` and ``BoolType``, dispatched through a
  single :func:`expr` entry point.
- One construction shape: a ``FuncDef`` with N real-typed parameters and a
  single ``ReturnStmt`` whose expression has type ``real``. (Bool sub-terms
  only appear inside ``IfExpr`` conditions and as the input/output of
  ``Compare`` / classification predicates — no bool-typed parameters yet.)
- Real ops: ``Integer`` literals, parameter ``Var`` references, arithmetic
  (``Add`` / ``Sub`` / ``Mul`` / ``Div`` / ``Neg`` / ``Abs``), a small set
  of named unary math ops (``Sqrt`` / ``Sin`` / ``Cos`` / ``Log`` / ``Exp``),
  and ``IfExpr``.
- Bool ops: ``BoolVal`` literals, ``Compare`` over real subexprs, real
  classification predicates (``IsFinite`` / ``IsNan`` / ``IsInf`` /
  ``IsNormal`` / ``Signbit``), and the boolean connectives ``Not`` /
  ``And`` / ``Or``.

Construction is AST-direct (we build ``FuncDef`` nodes and feed them into
the same passes the ``@fp.fpy`` decorator pipeline runs), so the parser is
*not* exercised here — the generator targets analyses and the interpreter.
"""

from typing import TypeAlias

from hypothesis import strategies as st

import fpy2 as fp

from fpy2.ast.fpyast import (
    Abs,
    Add,
    And,
    Argument,
    BoolVal,
    Compare,
    Cos,
    Div,
    Exp,
    Expr,
    FuncDef,
    FuncMeta,
    IfExpr,
    Integer,
    IsFinite,
    IsInf,
    IsNan,
    IsNormal,
    Log,
    Mul,
    Neg,
    NamedUnaryOp,
    Not,
    Or,
    RealTypeAnn,
    ReturnStmt,
    Signbit,
    Sin,
    Sqrt,
    StmtBlock,
    Sub,
    Var,
)
from fpy2.types import BoolType, RealType, Type
from fpy2.env import ForeignEnv
from fpy2.utils import CompareOp, NamedId


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TypeEnv: TypeAlias = dict[NamedId, Type]
"""Mapping from in-scope names to their FPy types."""


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

_LITERAL_RANGE = (-1_000, 1_000)


def _var_of_type(
    env: TypeEnv,
    target_cls: type[Type],
) -> st.SearchStrategy[Expr] | None:
    """Strategy for a ``Var`` of the given type class, or ``None`` if env has none."""
    names = sorted((n for n, t in env.items() if isinstance(t, target_cls)), key=str)
    if not names:
        return None
    return st.sampled_from(names).map(lambda n: Var(n, None))


def _func_sym(name: str) -> Var:
    """Construct the ``FuncSymbol`` used in ``Named*Op``.

    NamedUnaryOp etc. carry a ``func`` symbol that names the surface-syntax
    callable (e.g. ``sqrt``). Type inference doesn't look at this name (the
    type is determined by the AST class), but some passes do.
    """
    return Var(NamedId(name), None)


# ---------------------------------------------------------------------------
# Real-typed expressions
# ---------------------------------------------------------------------------

def _real_literal_strategy() -> st.SearchStrategy[Expr]:
    return st.integers(*_LITERAL_RANGE).map(lambda v: Integer(v, None))


# (AST class, surface-syntax name) — name matters for any pass that resolves
# the symbol back to a callable; the type-checker only looks at the class.
_REAL_NAMED_UNARY: list[tuple[type[NamedUnaryOp], str]] = [
    (Sqrt, 'sqrt'),
    (Sin, 'sin'),
    (Cos, 'cos'),
    (Log, 'log'),
    (Exp, 'exp'),
]


def _real_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``real`` under ``env``."""
    leaves: list[st.SearchStrategy[Expr]] = [_real_literal_strategy()]
    var_strat = _var_of_type(env, RealType)
    if var_strat is not None:
        leaves.append(var_strat)

    if depth <= 0:
        return st.one_of(*leaves)

    sub_real = _real_expr(env, depth - 1)
    sub_bool = _bool_expr(env, depth - 1)

    def _binop(cls):
        return st.tuples(sub_real, sub_real).map(lambda ab: cls(ab[0], ab[1], None))

    def _unop(cls):
        return sub_real.map(lambda x: cls(x, None))

    def _named_unary(cls, name):
        sym = _func_sym(name)
        return sub_real.map(lambda x: cls(sym, x, None))

    if_expr = st.tuples(sub_bool, sub_real, sub_real).map(
        lambda cab: IfExpr(cab[0], cab[1], cab[2], None)
    )

    inner: list[st.SearchStrategy[Expr]] = [
        _binop(Add),
        _binop(Sub),
        _binop(Mul),
        _binop(Div),
        _unop(Neg),
        _unop(Abs),
        if_expr,
    ]
    inner.extend(_named_unary(cls, name) for cls, name in _REAL_NAMED_UNARY)
    return st.one_of(*leaves, *inner)


# ---------------------------------------------------------------------------
# Bool-typed expressions
# ---------------------------------------------------------------------------

_COMPARE_OPS = list(CompareOp)

_REAL_PREDICATES: list[tuple[type[NamedUnaryOp], str]] = [
    (IsFinite, 'isfinite'),
    (IsInf, 'isinf'),
    (IsNan, 'isnan'),
    (IsNormal, 'isnormal'),
    (Signbit, 'signbit'),
]


def _bool_literal_strategy() -> st.SearchStrategy[Expr]:
    return st.booleans().map(lambda v: BoolVal(v, None))


def _bool_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``bool`` under ``env``."""
    leaves: list[st.SearchStrategy[Expr]] = [_bool_literal_strategy()]
    var_strat = _var_of_type(env, BoolType)
    if var_strat is not None:
        leaves.append(var_strat)

    if depth <= 0:
        return st.one_of(*leaves)

    sub_bool = _bool_expr(env, depth - 1)
    sub_real = _real_expr(env, depth - 1)

    # Compare: a chain of (n+1) reals connected by n comparison ops; restrict
    # to simple binary comparisons here (n = 1) — chains add little coverage
    # for what we're trying to fuzz.
    compare = st.tuples(
        st.sampled_from(_COMPARE_OPS),
        sub_real,
        sub_real,
    ).map(lambda oab: Compare([oab[0]], [oab[1], oab[2]], None))

    def _predicate(cls, name):
        sym = _func_sym(name)
        return sub_real.map(lambda x: cls(sym, x, None))

    not_ = sub_bool.map(lambda x: Not(x, None))
    and_or_args = st.lists(sub_bool, min_size=2, max_size=3)
    and_ = and_or_args.map(lambda xs: And(xs, None))
    or_ = and_or_args.map(lambda xs: Or(xs, None))

    inner: list[st.SearchStrategy[Expr]] = [compare, not_, and_, or_]
    inner.extend(_predicate(cls, name) for cls, name in _REAL_PREDICATES)
    return st.one_of(*leaves, *inner)


# ---------------------------------------------------------------------------
# Type-dispatched expression strategy
# ---------------------------------------------------------------------------

def expr(target: Type, env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Strategy for an expression of type ``target`` under ``env``.

    This is the canonical type-directed entry point. ``depth`` bounds the
    nesting depth of compound operations; at ``depth == 0`` only leaves
    (literals and variable references) are produced.

    Raises :class:`NotImplementedError` for type targets not yet supported
    (e.g. ``TupleType`` / ``ListType`` / ``ContextType``).
    """
    if isinstance(target, RealType):
        return _real_expr(env, depth)
    if isinstance(target, BoolType):
        return _bool_expr(env, depth)
    raise NotImplementedError(f'no generator for target type {target.format()}')


def real_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(RealType(), env, depth)``."""
    return _real_expr(env, depth)


def bool_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(BoolType(), env, depth)``."""
    return _bool_expr(env, depth)


# ---------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------

def _make_arg_names(n: int) -> list[NamedId]:
    return [NamedId(f'x{i}') for i in range(n)]


def _make_funcdef(name: str, arg_names: list[NamedId], body_expr: Expr) -> FuncDef:
    """Wrap an expression in a ``f(arg_names...) -> Real: return body_expr``."""
    args = [Argument(n, RealTypeAnn(None, None), None) for n in arg_names]
    body = StmtBlock([ReturnStmt(body_expr, None)])
    meta = FuncMeta(set(), None, None, {}, ForeignEnv.default())
    return FuncDef(name, args, body, meta)


@st.composite
def fpy_real_funcdef(
    draw,
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
) -> FuncDef:
    """Generate a well-typed FPy function ``f(x0..xN: Real) -> Real``.

    The body is a single ``return <expr>`` where ``<expr>`` is generated by
    :func:`real_expr` with the given depth budget.
    """
    if num_args is None:
        num_args = st.integers(0, 3)
    if max_depth is None:
        max_depth = st.integers(0, 3)

    n = draw(num_args)
    depth = draw(max_depth)

    arg_names = _make_arg_names(n)
    env: TypeEnv = {name: RealType() for name in arg_names}
    body_expr = draw(real_expr(env, depth))
    return _make_funcdef('f', arg_names, body_expr)


def fpy_real_function(
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
) -> st.SearchStrategy[fp.Function]:
    """Same as :func:`fpy_real_funcdef` but wraps the AST in :class:`fp.Function`."""
    return fpy_real_funcdef(num_args=num_args, max_depth=max_depth).map(fp.Function)
