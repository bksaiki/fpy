"""The routes by which a list's identity is observable.

A list compiles to ``fpy::list<T>`` — a shared handle — because FPy lists alias.
Representing one as a plain ``std::vector<T>`` (*unboxing*) would be free at a
native C++ boundary, where copying a nested list in and out costs ~8x, but it is
only sound when nothing can observe the difference between "this name shares an
object" and "this name owns a copy".

This module encodes the predicate for that and pins it against the alias
regressions in :mod:`tests.infra.backend.cpp`.  Nothing consumes the predicate
yet; it exists so the condition is written down and tested *before* a
representation change depends on it.

**Ground truth is measured, not argued.**  ``DIVERGED`` below records which
regressions gave a wrong answer when compiled with lists forced unboxed and
bit-compared against the interpreter.  Deriving it took four rounds, three of
which found *unsound* conditions:

- ``call_arg`` — boxedness must agree across a call edge, or the call site
  converts and the callee's write is lost.
- ``deep_slice`` — a slice is shallow: a fresh outer list over the *same*
  elements.  "A slice is fresh" is true of the outer list and false of what it
  holds.
- ``projection`` — ``row = xss[i]`` needs a reference binding; a value copy
  loses the write.  This one is a *codegen* obligation, not just a predicate
  entry, and it is the reason unboxing is more work than it looks.

The fourth round found ``returns_arg`` firing on a returned *scalar* element,
which cannot alias anything — over-conservative rather than unsound.

An AST-level condition cannot see a copy that codegen synthesizes, which is what
``synth_copy`` covers.
"""

import pytest

import fpy2 as fp
from fpy2.analysis import TypeInfer
from fpy2.ast.fpyast import (
    Assign,
    Call,
    ContextStmt,
    Enumerate,
    ForStmt,
    If1Stmt,
    IfStmt,
    IndexedAssign,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    NamedId,
    ReturnStmt,
    TupleExpr,
    Var,
    WhileStmt,
    Zip,
)
from fpy2.types import ListType

import tests.infra.backend.cpp as corpus

ROUTES = {
    'alias': 'ys = xs — the assignment copies the vector',
    'projection': 'row = xss[i] — needs a reference binding, not a value copy',
    'into_agg': '[xs] / (xs, ..) / a comprehension element — the container '
                'stores a copy',
    'slot_store': 'xss[i] = <shared list> — copies into the slot, and replaces '
                  'a slot a projection may reference',
    'returns_arg': "return <list rooted at a parameter> — the caller's binding "
                   'becomes a copy',
    'call_arg': 'g(xs) — boxedness must agree across the call edge',
    'deep_slice': 'xss[i:j] over lists — a slice is shallow, so the elements '
                  'are shared',
    'synth_copy': 'enumerate(xss) / zip(..) over lists — the lowering copies '
                  'each element, or turns it into a projection',
}


def _stmts(block):
    for stmt in block.stmts:
        yield stmt
        match stmt:
            case IfStmt():
                yield from _stmts(stmt.ift)
                yield from _stmts(stmt.iff)
            case If1Stmt() | WhileStmt() | ForStmt() | ContextStmt():
                yield from _stmts(stmt.body)


def _exprs(node):
    """Every expression reachable from *node*, statements included."""
    out = []

    def walk(e):
        if e is None or not hasattr(e, '__slots__'):
            return
        out.append(e)
        for cls in type(e).__mro__:
            for attr in getattr(cls, '__slots__', ()):
                v = getattr(e, attr, None)
                items = list(v) if isinstance(v, (list, tuple)) else [v]
                for item in items:
                    if hasattr(item, 'stmts'):
                        for st in item.stmts:
                            walk(st)
                    else:
                        walk(item)

    walk(node)
    return out


def _root_var(e):
    """The variable *e* names existing storage in, if any.

    A slice does *not* count: it materializes a new outer list.  Its elements are
    another matter — see ``deep_slice``.  The emitter draws the same distinction
    between ``_alias_root`` and ``_root_var``, and conflating them is an easy
    mistake: it made three ``fpy2.libraries.matrix`` functions look unsafe when
    their ``A[i][:]`` is a genuine copy.
    """
    while True:
        match e:
            case Var():
                return e
            case ListRef():
                e = e.value
            case _:
                return None


def alias_routes(func: fp.Function) -> set[str]:
    """The routes in *func* that make an unboxed list's identity observable.

    Empty means unboxing this function's lists is sound.  Deliberately a
    syntactic over-approximation: a route counts on shape alone, without proving
    the copy is really observed.  Over-approximating costs performance, never
    correctness — which is what makes unboxing safe to attempt at all.
    """
    ty = TypeInfer.check(func.ast)
    hits: set[str] = set()

    def is_list(e) -> bool:
        return isinstance(ty.by_expr.get(e), ListType)

    def shares(e) -> bool:
        """*e* denotes list storage that already exists elsewhere."""
        return is_list(e) and _root_var(e) is not None

    def nested(e) -> bool:
        t = ty.by_expr.get(e)
        return isinstance(t, ListType) and isinstance(t.elt, ListType)

    params = {
        str(a.name) for a in func.ast.args if isinstance(a.name, NamedId)
    }
    for stmt in _stmts(func.ast.body):
        match stmt:
            case Assign(expr=Var() as v) if is_list(v):
                hits.add('alias')
            case Assign(expr=ListRef() as r) if is_list(r):
                hits.add('projection')
            case IndexedAssign() if shares(stmt.expr):
                hits.add('slot_store')
            case ReturnStmt() if is_list(stmt.expr):
                root = _root_var(stmt.expr)
                if root is not None and str(root.name) in params:
                    hits.add('returns_arg')

    for e in _exprs(func.ast):
        match e:
            case Call() if any(is_list(a) for a in e.args):
                hits.add('call_arg')
            case ListSlice() if nested(e):
                hits.add('deep_slice')
            case Enumerate() | Zip():
                if any(nested(a) for a in e.args):
                    hits.add('synth_copy')
            case ListExpr() | TupleExpr():
                if any(shares(x) for x in e.elts):
                    hits.add('into_agg')
            case ListComp() if shares(e.elt):
                hits.add('into_agg')
    return hits


# Measured: compiled with lists forced unboxed, these gave a *wrong answer*
# against the interpreter.  Each must therefore be rejected by the predicate.
DIVERGED = {
    '_regression_returned_list_aliases',
    '_regression_projected_element',
    '_regression_list_into_list',
    '_regression_list_into_tuple',
    '_regression_comprehension_of_rows',
    '_regression_nested_slice',
    '_regression_one_list_two_indices',
    # Still diverges after `EnumerateElim` (#235): the elimination turns the
    # synthesized tuple copy into a projection, which needs a reference binding.
    '_regression_enumerate_row_write',
    # Rejected by the C++ compiler rather than mis-executed, which is a
    # divergence all the same.
    '_regression_alias_then_mutate',
}

# Measured: still bit-matched the interpreter with lists unboxed.  Those with an
# empty route set below are where the predicate has to be *precise* — rejecting
# everything would be sound and useless.
AGREED = {
    '_regression_alias_readonly',
    '_regression_callee_mutates_param',
    '_regression_writes_its_arg',
    '_regression_loop_variable',
    '_regression_comprehension_deep_copy',
    '_regression_flat_slice',
    '_regression_conditional_alias',
    '_regression_replaced_slot',
}

# The predicate's verdict per program.  Where a program is in AGREED but has a
# non-empty set, the predicate is over-conservative: sound, just less precise
# than it could be.
EXPECTED: dict[str, set[str]] = {
    '_regression_alias_then_mutate': {'alias'},
    '_regression_alias_readonly': {'alias'},
    '_regression_callee_mutates_param': {'call_arg'},
    '_regression_writes_its_arg': set(),
    '_regression_returns_its_argument': {'returns_arg'},
    '_regression_returned_list_aliases': {'call_arg'},
    '_regression_projected_element': {'projection'},
    '_regression_loop_variable': set(),
    '_regression_list_into_list': {'into_agg'},
    '_regression_list_into_tuple': {'into_agg'},
    '_regression_comprehension_of_rows': {'into_agg'},
    '_regression_comprehension_deep_copy': set(),
    '_regression_nested_slice': {'deep_slice'},
    '_regression_flat_slice': set(),
    '_regression_one_list_two_indices': {'into_agg'},
    '_regression_enumerate_row_write': {'synth_copy'},
    '_regression_conditional_alias': {'alias'},
    '_regression_replaced_slot': {'projection', 'slot_store'},
}


def _func(name: str) -> fp.Function:
    f = getattr(corpus, name, None)
    assert isinstance(f, fp.Function), f'{name} is not in the corpus'
    return f


class TestAliasRoutes:
    """Soundness first: everything measured to diverge must be rejected.
    Precision second: the cases measured to agree should ideally be accepted."""

    @pytest.mark.parametrize('name', sorted(DIVERGED))
    def test_diverging_programs_are_rejected(self, name):
        """The load-bearing assertion.

        Each of these produced a wrong answer (or failed to compile) with lists
        unboxed, so a predicate that approves one would authorise a
        miscompilation.
        """
        assert alias_routes(_func(name)), (
            f'{name} diverges when unboxed but the predicate approves it'
        )

    @pytest.mark.parametrize(
        'name', sorted(n for n in AGREED if not EXPECTED[n]),
    )
    def test_unobservable_copies_are_accepted(self, name):
        """Where the predicate must be precise, not merely conservative.

        FPy copies at these sites too — a fresh comprehension element, a flat
        slice, an element write through a loop variable — so C++ copying is not a
        divergence and rejecting them would forgo the whole benefit.
        """
        assert alias_routes(_func(name)) == set()

    @pytest.mark.parametrize('name', sorted(EXPECTED), ids=lambda n: n[12:])
    def test_verdict_is_unchanged(self, name):
        """Pins the exact verdict, so a predicate change is visible in review
        rather than silently shifting what gets unboxed."""
        assert alias_routes(_func(name)) == EXPECTED[name]

    def test_ground_truth_covers_every_verdict(self):
        assert set(EXPECTED) - {'_regression_returns_its_argument'} == (
            DIVERGED | AGREED
        ), 'every classified program needs a measured outcome'
        assert not (DIVERGED & AGREED)

    def test_every_flagged_regression_has_a_verdict(self):
        """A corpus regression the predicate flags must have ground truth.

        Inverted deliberately: rather than guess from names which regressions are
        alias-relevant, let the predicate say.  A new one that trips a route then
        has to be compiled unboxed and recorded here.
        """
        flagged = {f.name for f in corpus._regression_funcs if alias_routes(f)}
        assert not flagged - set(EXPECTED), (
            'these trip an aliasing route but have no measured outcome; compile '
            f'them unboxed and record it: {sorted(flagged - set(EXPECTED))}'
        )

    def test_routes_are_documented_and_live(self):
        """Every route is documented, and every documented route is exercised."""
        reported: set[str] = set()
        for name in EXPECTED:
            reported |= alias_routes(_func(name))
        assert reported <= set(ROUTES), reported - set(ROUTES)
        assert set(ROUTES) <= reported, set(ROUTES) - reported
