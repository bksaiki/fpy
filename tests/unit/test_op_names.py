"""
Tests for `CANONICAL_OP_NAMES` (fpy2/ast/fpyast.py).

These tie the canonical node-type -> name registry to the parser's
name -> node-type tables so the two cannot drift apart: the pretty-printer
uses `CANONICAL_OP_NAMES` to render an operator synthesized by a rewrite
pass (whose `func` symbol is `None`), and that rendered name must resolve
back to the same operator when re-parsed.
"""

import builtins
import inspect

import fpy2
import fpy2.frontend.parser as parser

from fpy2.ast import fpyast
from fpy2.ast.fpyast import (
    CANONICAL_OP_NAMES,
    NullaryOp, NamedUnaryOp, NamedBinaryOp, NamedTernaryOp, NamedNaryOp,
    Hexnum, Rational, Digits,
)

# every node class whose `func` symbol may be `None` (and therefore must be
# renderable from its type alone)
_FUNC_BEARING_BASES = (
    NullaryOp, NamedUnaryOp, NamedBinaryOp, NamedTernaryOp, NamedNaryOp,
)

_PARSER_TABLES = (
    parser._nullary_table,
    parser._unary_table,
    parser._binary_table,
    parser._ternary_table,
    parser._nary_table,
)


def _concrete_func_bearing_classes() -> set[type]:
    classes = {
        obj
        for obj in vars(fpyast).values()
        if inspect.isclass(obj)
        and issubclass(obj, _FUNC_BEARING_BASES)
        and obj not in _FUNC_BEARING_BASES
    }
    classes |= {Hexnum, Rational, Digits}
    return classes


def _parser_class_to_callables() -> dict[type, set]:
    m: dict[type, set] = {}
    for table in _PARSER_TABLES:
        for fn, cls in table.items():
            m.setdefault(cls, set()).add(fn)
    return m


def _resolve(name: str):
    """The callable a canonical name resolves to, preferring fpy2 over builtins
    (the parser resolves `fp.<name>` before a bare builtin)."""
    obj = getattr(fpy2, name, None)
    if obj is None:
        obj = getattr(builtins, name, None)
    return obj


def test_registry_covers_all_func_bearing_classes():
    missing = sorted(
        c.__name__ for c in _concrete_func_bearing_classes() - set(CANONICAL_OP_NAMES)
    )
    assert not missing, f'operator classes with no canonical name: {missing}'


def test_registry_has_no_stale_entries():
    stale = sorted(
        c.__name__ for c in set(CANONICAL_OP_NAMES) - _concrete_func_bearing_classes()
    )
    assert not stale, f'canonical names for non-operator classes: {stale}'


def test_canonical_names_resolve():
    for cls, name in CANONICAL_OP_NAMES.items():
        assert _resolve(name) is not None, (
            f'{cls.__name__}: canonical name {name!r} is neither an fpy2 '
            f'attribute nor a Python builtin'
        )


def test_canonical_names_map_back_to_class():
    # for every class the parser produces from a surface name, that name must
    # resolve to a callable the parser maps back to the same class
    for cls, callables in _parser_class_to_callables().items():
        if cls not in CANONICAL_OP_NAMES:
            continue  # non-named ops (Add, Abs, ...) render without a func symbol
        name = CANONICAL_OP_NAMES[cls]
        resolved = _resolve(name)
        assert resolved in callables, (
            f'{cls.__name__}: canonical name {name!r} resolves to a callable '
            f'the parser does not map to {cls.__name__}'
        )
