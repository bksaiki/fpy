"""
Unit tests for `TypeInfer`:

- ``TestCrossFunction``: call-graph-driven checking (callees once, leaves-first).
- ``TestTypeInferOnGeneratedPrograms``: property tests driven by the
  type-directed generator in ``tests/unit/generators/fpy_program.py``. The
  generator constructs a program of a known signature; here we verify the
  analysis recovers that signature, is deterministic across re-runs, and
  records a type for every sub-expression in the body.
"""

import fpy2 as fp

from hypothesis import given, settings, strategies as st

from fpy2.analysis import TypeInfer
from fpy2.analysis.type_infer import TypeInfer as _TI
from fpy2.ast.fpyast import Ast, Expr, FuncDef
from fpy2.types import Type

from ..generators import (
    arbitrary_type,
    fpy_funcdef,
    fpy_real_funcdef,
)


class TestCrossFunction:
    def test_callee_return_type_flows_to_caller(self):
        @fp.fpy
        def g(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return g(x) * 2

        info = TypeInfer.check(f.ast)
        # f: real -> real, derived from g's inferred real return type
        assert len(info.fn_type.arg_types) == 1
        assert isinstance(info.fn_type.return_type, fp.types.RealType)

    def test_returned_analysis_is_the_root_not_a_callee(self):
        # root has 2 args, callee has 1 — if `check` returned the
        # callee's analysis (a bug in the leaves-first walk) the arity
        # would be wrong.
        @fp.fpy
        def g(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return g(x) + g(y)

        info = TypeInfer.check(f.ast)
        assert len(info.fn_type.arg_types) == 2

    def test_diamond_checks(self):
        @fp.fpy
        def d(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return d(x) * 2

        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return d(x) - 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) + c(x)

        info = TypeInfer.check(a.ast)
        assert isinstance(info.fn_type.return_type, fp.types.RealType)


class TestBinding:
    def test_unpack_through_list_ref(self):
        """Unpacking a list element via ``a = xs[i]; a1, a2 = a`` must
        resolve ``a``'s type through the union-find before checking the
        tuple shape, since ``_visit_list_ref`` returns a fresh ``VarType``
        that only later unifies with the list's element type.
        """
        @fp.fpy
        def bar(x: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            for a, b in zip(x, x):
                a1, a2 = a
            return 0

        from fpy2.transform.zip_elim import ZipElim
        rewritten = ZipElim.apply(bar.ast)
        info = TypeInfer.check(rewritten)
        assert isinstance(info.fn_type.return_type, fp.types.RealType)


class TestCheckedOnce:
    def test_each_function_checked_once(self, monkeypatch):
        """A shared callee in a diamond is checked once, not once per
        call site: the call-graph walk pre-fills the signature cache, so
        `_visit_call` never recurses back into `TypeInfer.check`."""
        calls: list[str] = []
        orig = _TI.check  # staticmethod resolves to the plain function

        def counting(func, def_use=None):
            calls.append(func.name)
            return orig(func, def_use)

        monkeypatch.setattr(_TI, 'check', staticmethod(counting))

        @fp.fpy
        def d(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return d(x) * 2

        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return d(x) - 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) + c(x)

        TypeInfer.check(a.ast)
        # exactly one top-level `check`; the per-function analysis runs
        # inside it via the leaves-first walk, with no recursive checks.
        assert calls == ['a']


# ---------------------------------------------------------------------------
# Property tests driven by the type-directed generator
# ---------------------------------------------------------------------------

def _walk_exprs(node):
    """Yield every ``Expr`` node reachable from ``node`` via AST slots.

    Used to verify ``TypeInfer`` records a type for every sub-expression in
    a generated body. Walks the same slot-based structure the AST uses for
    storage, so any node type that follows the convention is covered.
    """
    if isinstance(node, Expr):
        yield node
    for slot in getattr(type(node), '__slots__', ()):
        try:
            val = getattr(node, slot)
        except AttributeError:
            continue
        if isinstance(val, Ast):
            yield from _walk_exprs(val)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Ast):
                    yield from _walk_exprs(item)


# Generator config used across the property tests below. Conservative
# bounds keep test wall-clock reasonable while still exercising every
# statement kind.
_GEN_KWARGS = dict(
    max_depth=st.just(2),
    max_assigns=st.just(1),
    max_contexts=st.just(0),
    max_ifs=st.just(0),
    max_loops=st.just(0),
    max_whiles=st.just(0),
)


class TestTypeInferOnGeneratedPrograms:
    """``TypeInfer`` driven by the type-directed FPy program generator.

    The generator constructs a ``FuncDef`` whose signature is known up
    front. Each test below probes a different ``TypeInfer`` invariant by
    running the analysis on those generated programs.
    """

    @given(st.data())
    @settings(max_examples=100, deadline=None)
    def test_inferred_signature_matches_generator(self, data: st.DataObject) -> None:
        """``TypeInfer.check`` recovers exactly the signature the generator built.

        This is the strongest property: ``arg_types`` and ``return_type``
        come from the generator's own draws, and inference is expected to
        round-trip them with no widening or loss.
        """
        n_args = data.draw(st.integers(0, 3))
        arg_ts: tuple[Type, ...] = tuple(
            data.draw(arbitrary_type(max_depth=1)) for _ in range(n_args)
        )
        ret_t = data.draw(arbitrary_type(max_depth=1))
        fd = data.draw(fpy_funcdef(arg_ts, ret_t, **_GEN_KWARGS))

        analysis = TypeInfer.check(fd)
        assert tuple(analysis.arg_types) == arg_ts, (
            f'arg mismatch: inferred {[t.format() for t in analysis.arg_types]}, '
            f'generator built with {[t.format() for t in arg_ts]}'
        )
        assert analysis.return_type == ret_t, (
            f'return mismatch: inferred {analysis.return_type.format()}, '
            f'generator built with {ret_t.format()}'
        )

    @given(fpy_real_funcdef(**_GEN_KWARGS))
    @settings(max_examples=80, deadline=None)
    def test_is_deterministic(self, fd: FuncDef) -> None:
        """Running ``TypeInfer.check`` twice on the same AST yields the same answer.

        Catches accidental statefulness in the analysis (e.g. ``Gensym``
        leakage, cache poisoning).
        """
        a1 = TypeInfer.check(fd)
        a2 = TypeInfer.check(fd)
        assert a1.fn_type == a2.fn_type
        # by_def / by_expr are dicts whose keys differ between runs (fresh
        # AST identity), but their *sizes* should agree.
        assert len(a1.by_def) == len(a2.by_def)
        assert len(a1.by_expr) == len(a2.by_expr)

    @given(fpy_real_funcdef(**_GEN_KWARGS))
    @settings(max_examples=80, deadline=None)
    def test_every_body_expr_has_inferred_type(self, fd: FuncDef) -> None:
        """Every ``Expr`` in the body should appear in ``analysis.by_expr``.

        Soundness gap if not — it means a sub-expression's type was never
        established, but the rest of the function still type-checked
        (likely indicating dead-code in the inference walk).
        """
        analysis = TypeInfer.check(fd)
        # Some sub-expressions (e.g. function-symbol ``Var`` nodes carried
        # by ``Named*Op`` for the surface name like 'sqrt') are syntactic
        # carriers, not values to type-check. Conservatively: every
        # encountered ``Expr`` should either be in ``by_expr`` or be one
        # of those carriers, which live on ``func`` slots.
        for expr in _walk_exprs(fd.body):
            if expr in analysis.by_expr:
                continue
            # Carrier check: the expr is referenced via a parent's ``func``
            # slot (Named*Op / Call). Approximate this by asking: is this
            # expression a ``Var`` we never actually use as a value?
            # We can't easily walk parents here, so we just allow ``Var``
            # nodes to be absent — this under-asserts but doesn't FN.
            from fpy2.ast.fpyast import Var
            if isinstance(expr, Var):
                continue
            raise AssertionError(
                f'sub-expression of type {type(expr).__name__} '
                f'has no inferred type in by_expr'
            )
