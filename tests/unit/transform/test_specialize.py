"""
Unit tests for module-level specialization.

A spec is identified by its function, its calling context, its refined
argument *types*, and the argument *values* the caller pinned.  The last two
are separate axes: a type says what a value may be, a pin says which value it
is.
"""

import fpy2 as fp
from fpy2.transform import Specialize
from fpy2.types import ListType, RealType

_ROUND_AT = fp.MPFixedContext(-4, fp.RM.RTZ)


def _specs(module: fp.Module, *, size_key: bool = False) -> dict[str, fp.Function]:
    """Every spec of *module*, by name."""
    out = Specialize.apply(module, size_key=size_key)
    return {f.name: f for f in out.functions()}


def _module(*entries) -> fp.Module:
    m = fp.Module()
    for func, arg_types in entries:
        m.add(func, arg_types=arg_types)
    return m


class TestPinnedValues:
    """A context or rounding mode the caller knows is substituted into the
    callee, which is what makes a `with` over a *parameter* resolvable."""

    def test_a_pinned_context_reaches_the_body(self):
        @fp.fpy(ctx=fp.REAL)
        def callee(x, rho):
            with rho:
                return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def caller(x):
            return callee(x, fp.FP32)

        specs = _specs(_module((caller, [RealType(fp.FP64)])))
        body = next(f for n, f in specs.items() if n.startswith('callee')).format()
        assert 'with rho:' not in body
        assert 'FP32' in body

    def test_a_pinned_rounding_mode_reaches_the_body(self):
        """A rounding mode has no format, so it can only travel as a value."""
        @fp.fpy(ctx=fp.REAL)
        def callee(x, rm):
            with fp.MPFixedContext(-4, rm):
                return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def caller(x):
            return callee(x, fp.RM.RTZ)

        specs = _specs(_module((caller, [RealType(fp.FP64)])))
        body = next(f for n, f in specs.items() if n.startswith('callee')).format()
        assert 'RoundingMode.RTZ' in body

    def test_two_pins_of_one_callee_are_two_specs(self):
        @fp.fpy(ctx=fp.REAL)
        def callee(x, rho):
            with rho:
                return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def caller(x):
            return callee(x, fp.FP32) + callee(x, fp.FP16)

        specs = _specs(_module((caller, [RealType(fp.FP64)])))
        callees = [n for n in specs if n.startswith('callee')]
        assert len(callees) == 2, callees

    def test_one_pin_shared_by_two_sites_is_one_spec(self):
        @fp.fpy(ctx=fp.REAL)
        def callee(x, rho):
            with rho:
                return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def caller(x):
            return callee(x, fp.FP32) + callee(x, fp.FP32)

        specs = _specs(_module((caller, [RealType(fp.FP64)])))
        assert len([n for n in specs if n.startswith('callee')]) == 1


class TestArgumentTypes:
    """The refined argument types are the other axis, and are exactly what
    `Monomorphize` is given -- so one fingerprint means one body."""

    def test_two_formats_are_two_specs(self):
        @fp.fpy(ctx=fp.REAL)
        def callee(x):
            with fp.FP32:
                return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def caller(x, y):
            return callee(x) + callee(y)

        specs = _specs(_module((caller, [RealType(fp.FP16), RealType(fp.FP64)])))
        assert len([n for n in specs if n.startswith('callee')]) == 2

    def test_lengths_key_only_under_size_key(self):
        @fp.fpy(ctx=fp.REAL)
        def callee(xs):
            return xs[0]

        @fp.fpy(ctx=fp.REAL)
        def caller(xs, ys):
            return callee(xs) + callee(ys)

        mod = _module((caller, [
            ListType(RealType(fp.FP32), 2), ListType(RealType(fp.FP32), 3),
        ]))
        assert len([n for n in _specs(mod) if n.startswith('callee')]) == 1
        assert len([
            n for n in _specs(mod, size_key=True) if n.startswith('callee')
        ]) == 2


class TestStability:
    """Spec names reach generated code, so they must not move between runs."""

    def test_names_are_stable_across_runs(self):
        @fp.fpy(ctx=fp.REAL)
        def callee(x, rho):
            with rho:
                return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def caller(xs):
            return callee(xs[0], _ROUND_AT)

        args = [ListType(RealType(fp.FP32), 4)]
        first = sorted(_specs(_module((caller, args)), size_key=True))
        second = sorted(_specs(_module((caller, args)), size_key=True))
        assert first == second


class TestShape:
    """An argument's *shape* pins even when its leaves do not."""

    def test_a_list_argument_is_not_the_polymorphic_spec(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            return xs[0]

        for order in (('listy', 'poly'), ('poly', 'listy')):
            mod = fp.Module()
            for name in order:
                mod.add(f, name=name, arg_types=(
                    [ListType(RealType(None), 8)] if name == 'listy' else None
                ))
            assert sorted(_specs(mod)) == ['listy', 'poly'], order

    def test_a_discarded_parameter_does_not_split_the_spec(self):
        @fp.fpy(ctx=fp.REAL)
        def callee(x, _):
            return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def caller(x):
            return callee(x, fp.FP32) + callee(x, fp.FP16) + callee(x, fp.FP64)

        specs = _specs(_module((caller, [RealType(fp.FP64)])))
        assert len([n for n in specs if n.startswith('callee')]) == 1


class TestAUserNameIsNotUnmangled:
    """Specialization re-reads its own output, so it has to recover a spec's
    base name -- but by tracking what it coined, not by stripping a suffix a
    user's name can have too."""

    def test_two_functions_keep_two_symbols(self):
        import fpy2 as fp
        from fpy2.backend.cpp import CppCompiler
        from fpy2.types import RealType

        @fp.fpy(ctx=fp.FP64)
        def helper__deadbeef(x):
            return x + 1

        @fp.fpy(ctx=fp.FP64)
        def helper(x):
            return x + 2

        @fp.fpy(ctx=fp.FP64)
        def entry(x):
            return helper__deadbeef(x) + helper(x)

        m = fp.Module()
        m.add(entry, arg_types=[RealType(fp.FP64)])
        out = CppCompiler().compile_module(m)
        defs = [l for l in out.splitlines() if l.startswith('double helper')]
        assert len(defs) == 2 and len(set(defs)) == 2, defs
