"""``UnboxMode.STRICT``: unbox everywhere or refuse to compile.

``ALLOW``'s contract is per-level best effort; ``STRICT``'s is a guarantee --
the emitted unit contains no ``std::shared_ptr``.  A program
whose semantics need a shared handle is a compile error naming the list and
the reason it kept its handle, not a handle in the output.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp.compiler import CppCompileError, CppCompiler
from fpy2.backend.cpp.unbox import UnboxMode
from fpy2.module import Module
from fpy2.types import BoolType, ListType, RealType

R = RealType(fp.FP64)
L = ListType(R)

STRICT = CppCompiler(unbox=UnboxMode.STRICT)


@fp.fpy
def _scale(xs: list[fp.Real], a: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        return [a * x for x in xs]


@fp.fpy
def _shared(xs: list[fp.Real], c: bool, x: fp.Real) -> fp.Real:
    """`ys` may be `xs` or a fresh list, and `ys[0] = 99` must reach `xs`
    on one path -- the canonical program only a handle can compile."""
    with fp.FP64:
        if c:
            ys = [x, x]
        else:
            ys = xs
        ys[0] = 99
        return xs[0]


@fp.fpy
def _identity(xs: list[fp.Real]) -> list[fp.Real]:
    """Returning a parameter hands the caller a second name for its own
    storage -- unboxed, the return would be a copy and lose writes."""
    with fp.FP64:
        return xs


class TestStrictAccepts:
    """Where every level unboxes, STRICT is exactly ALLOW."""

    def test_fully_unboxable_program_compiles_clean(self):
        out = STRICT.compile(_scale, ctx=fp.FP64, arg_types=[L, R])
        assert 'std::vector<double>' in out
        assert 'std::shared_ptr' not in out

    def test_signature_reports_native_types(self):
        params, ret = STRICT.signature(_scale, ctx=fp.FP64, arg_types=[L, R])
        assert params[0].format() == 'std::vector<double>'
        assert ret.format() == 'std::vector<double>'

    def test_signature_is_not_poisoned_by_an_unrelated_function(self):
        """An unrelated function's strict refusal is not the entry's problem,
        and must not become one by way of `Module.add` order."""
        for order in ([_shared, _scale], [_scale, _shared]):
            m = Module()
            for f in order:
                if f is _shared:
                    m.add(f, ctx=fp.FP64, arg_types=[L, BoolType(), R])
                else:
                    m.add(f, ctx=fp.FP64, arg_types=[L, R])
            params, ret = STRICT.signature(
                _scale, ctx=fp.FP64, arg_types=[L, R], module=m,
            )
            assert params[0].format() == 'std::vector<double>'
            assert ret.format() == 'std::vector<double>'


class TestStrictRefuses:
    """Where a list must keep its handle, compilation fails -- naming the
    list, the reason, and the way out."""

    def test_the_error_names_the_list_reason_and_escape_hatch(self):
        with pytest.raises(
            CppCompileError,
            match=r'(?s)`ys` \(depth 0\): shared.*UnboxMode\.ALLOW',
        ):
            STRICT.compile(
                _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
            )

    def test_returned_parameter_is_one_offender_not_two(self):
        """A returned parameter is *one* list: it is named as `xs`, and the
        `<return>` entry for the same region is deduplicated away."""
        with pytest.raises(CppCompileError, match=r'`xs`') as exc:
            STRICT.compile(_identity, ctx=fp.FP64, arg_types=[L])
        assert '<return>' not in str(exc.value), str(exc.value)

    def test_an_unnamed_returned_list_reports_as_return(self):
        """A `return` joining a parameter with a fresh literal boxes both;
        the literal is bound to no name, so only `<return>` can report it."""
        @fp.fpy
        def f(xs: list[fp.Real], c: bool, y: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                if c:
                    return xs
                return [y, y]

        with pytest.raises(CppCompileError, match=r'<return> \(depth 0\)'):
            STRICT.compile(f, ctx=fp.FP64, arg_types=[L, BoolType(), R])

    def test_a_list_shared_through_a_tuple_field_is_refused(self):
        """The tuple-field branch of the walk: `_stamp` records no reason
        down a tuple field, so the report falls back to naming the shape."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = (xs, 1.0)
                w = fp.fst(t)
                w[0] = 55
                return xs[0]

        with pytest.raises(
            CppCompileError, match=r'`t` \(depth 0\): shared \(tuple field\)',
        ):
            STRICT.compile(f, ctx=fp.FP64, arg_types=[L])

    def test_a_retaining_callee_fails_strict(self):
        """A callee that keeps its argument forces handles on both ends of
        the call -- leaves-first, the callee is refused before the caller
        is ever emitted, so the error names `keep` and the boundary."""
        @fp.fpy
        def keep(zs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return zs

        @fp.fpy
        def hand_over(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = keep(xs)
                return ys[0]

        m = Module()
        m.add(hand_over, ctx=fp.FP64, arg_types=[L])
        with pytest.raises(
            CppCompileError,
            match=r'(?s)strict unboxing failed for `keep'
                  r'.*reached across a boundary',
        ):
            STRICT.compile_module(m)

    def test_signature_is_strict_too(self):
        """`signature` goes through the same `analyze`, so an embedding
        program cannot be told types that `compile_module` would refuse."""
        with pytest.raises(CppCompileError, match='strict unboxing failed'):
            STRICT.signature(
                _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
            )

    def test_an_unstorable_return_is_not_blamed_on_strictness(self):
        """A REAL-format return fails storage selection in every mode; under
        STRICT it must present as that failure, not as a strict refusal."""
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.REAL:
                return x * y

        with pytest.raises(
            CppCompileError, match='storage selection failed',
        ) as exc:
            STRICT.compile(f, ctx=fp.FP64, arg_types=[R, R])
        assert 'strict unboxing' not in str(exc.value)


class TestEmitterTripwire:
    """The emitter's `_is_boxed` tripwire is the last line of defense: every
    handle spelling branches on it, so a boxed type that survives both
    analysis layers is refused at the point the handle would become real.
    No real program can get past those layers (that is their contract), so
    the regression is simulated by disabling them."""

    def test_tripwire_fires_if_both_analysis_layers_regress(self, monkeypatch):
        import fpy2.backend.cpp.compiler as compiler_mod
        from fpy2.backend.cpp.unbox import UnboxAnalysis

        def lenient_annotate(self, e, ty):
            def at(depth):
                r = self.alias.region_of_expr(e, depth)
                return set() if r is None else {r}
            return self._stamp(ty, at, 0)

        monkeypatch.setattr(compiler_mod, 'check_strict', lambda *a: None)
        monkeypatch.setattr(UnboxAnalysis, 'annotate', lenient_annotate)
        with pytest.raises(
            CppCompileError, match='internal error.*shared handle',
        ):
            STRICT.compile(
                _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
            )


class TestStrictIsTheDefault:
    """A bare ``CppCompiler()`` refuses a handle rather than emitting one."""

    def test_default_refuses_a_shared_list(self):
        with pytest.raises(CppCompileError, match='strict unboxing failed'):
            CppCompiler().compile(
                _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
            )

    def test_default_compiles_an_unboxable_kernel(self):
        assert 'std::shared_ptr' not in CppCompiler().compile(
            _scale, ctx=fp.FP64, arg_types=[L, R],
        )


class TestOtherModesUnchanged:
    """STRICT is a third mode, not a change to the other two."""

    @pytest.mark.parametrize(
        'mode', [UnboxMode.ALLOW, UnboxMode.NEVER], ids=['allow', 'never'],
    )
    def test_other_modes_still_compile_shared_programs(self, mode):
        out = CppCompiler(unbox=mode).compile(
            _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
        )
        assert 'std::shared_ptr<std::vector<double>>' in out
