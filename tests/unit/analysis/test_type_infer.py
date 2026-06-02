"""
Unit tests for `TypeInfer`, focused on the call-graph-driven checking:
callees are checked once in leaves-first order and their signatures are
reused at call sites, rather than re-checked per site.
"""

import fpy2 as fp

from fpy2.analysis import TypeInfer
from fpy2.analysis.type_infer import TypeInfer as _TI


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
        """Unpacking a list element directly — ``a = xs[i]; a1, a2 = a``
        — must resolve ``a``'s type through the union-find before
        checking the tuple shape.

        Regression: ``_visit_list_ref`` returns a fresh ``VarType``
        that ``_unify`` links to the list's element type.  Without
        resolving in ``_visit_binding``, the tuple-binding case saw a
        ``VarType`` (not a ``TupleType``) and raised ``cannot unpack``.
        Triggered through the cpp ``optimize=True`` path: ``ZipElim``
        rewrites ``for a, b in zip(xs, xs): ...`` into ``a = _src[_i]``,
        then ``a1, a2 = a`` failed to type-check.
        """
        @fp.fpy
        def bar(x: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            for a, b in zip(x, x):
                a1, a2 = a
            return 0

        from fpy2.transform.zip_elim import ZipElim
        rewritten = ZipElim.apply(bar.ast)
        info = TypeInfer.check(rewritten)  # raised before the fix
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
