"""
Unit tests for module-level specialization.

A spec is identified by its function, its calling context, its refined
argument *types*, the argument *values* the caller pinned, and the bounds the
caller's analysis derived inside it.  Types and values are separate axes: a
type says what a value may be, a pin says which value it is.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

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
    `Monomorphize` is given -- so one key means one body."""

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

    def test_names_are_stable_across_processes(self):
        """Spec names reach generated code, so they must not depend on hash
        randomization -- which an analysis keyed on AST node identity will
        leak into any iteration order it exposes."""
        prog = textwrap.dedent("""
            import fpy2 as fp
            from fpy2.transform import Specialize
            from fpy2.types import ListType, RealType

            # branches in a loop, so the callee has phi definitions -- which
            # is what makes `by_def` order follow node identity
            @fp.fpy(ctx=fp.REAL)
            def scan(xs, n):
                hit = False
                acc = 0
                with fp.MPFixedContext(n, fp.RM.RTZ):
                    ts = [fp.round(x) for x in xs]
                for t in ts:
                    if t > 0:
                        hit = True
                        acc = acc + t
                    else:
                        acc = acc - t
                return acc if hit else -acc

            @fp.fpy(ctx=fp.REAL)
            def caller(xs, ys):
                ps = [x * y for x, y in zip(xs, ys)]
                e = max([fp.logb(p) for p in ps])
                return scan(ps, e - 12)

            mod = fp.Module()
            mod.add(caller, arg_types=[ListType(RealType(fp.FP16), 8)] * 2)
            print(sorted(f.name for f in Specialize.apply(mod).functions()))
        """)
        # from a file, not `python -c`: the decorator reads the function's
        # source, which only 3.13 keeps for a `-c` command
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'prog.py')
            with open(path, 'w') as f:
                f.write(prog)
            runs = {
                subprocess.run(
                    [sys.executable, path], capture_output=True, text=True,
                    check=True, env={**os.environ, 'PYTHONHASHSEED': seed},
                ).stdout
                for seed in ('0', '1', '12345')
            }
        assert len(runs) == 1, runs

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


class TestDerivedBounds:
    """A caller's relation between two arguments -- `n` is `xs`'s greatest
    exponent less twelve -- reduces to no per-argument format, so it cannot be
    keyed directly.  The bounds it *yields* inside the callee can be."""

    @staticmethod
    def _rnd():
        @fp.fpy(ctx=fp.REAL)
        def rnd(xs, n):
            with fp.MPFixedContext(n, fp.RM.RTZ):
                ts = [fp.round(x) for x in xs]
            return sum(ts)
        return rnd

    def test_callers_that_bound_differently_get_separate_specs(self):
        """`e` bounds `ps`'s exponents and says nothing about `qs`, so the two
        calls need different precisions -- 12 and 22.  Argument formats and
        values are identical, so nothing else in the key separates them."""
        rnd = self._rnd()

        @fp.fpy(ctx=fp.REAL)
        def caller(xs, ys, zs, ws):
            ps = [x * y for x, y in zip(xs, ys)]
            qs = [z * w for z, w in zip(zs, ws)]
            e = max([fp.logb(p) for p in ps])
            return rnd(ps, e - 12) + rnd(qs, e - 12)

        l8 = ListType(RealType(fp.FP16), 8)
        specs = _specs(_module((caller, [l8, l8, l8, l8])))
        assert len([n for n in specs if n.startswith('rnd')]) == 2

    def test_one_relation_keeps_one_spec(self):
        """The same relation twice derives the same bounds, so the callers
        share a spec rather than each taking a copy."""
        rnd = self._rnd()

        @fp.fpy(ctx=fp.REAL)
        def caller(xs, ys):
            ps = [x * y for x, y in zip(xs, ys)]
            e = max([fp.logb(p) for p in ps])
            return rnd(ps, e - 12) + rnd(ps, e - 12)

        l8 = ListType(RealType(fp.FP16), 8)
        specs = _specs(_module((caller, [l8, l8])))
        assert len([n for n in specs if n.startswith('rnd')]) == 1

    def test_trivial_bounds_add_no_key_segment(self):
        """A callee the caller bounds at nothing is named as it was before the
        axis existed -- here one segment, for the calling context alone."""
        @fp.fpy(ctx=fp.REAL)
        def leaf(x):
            return x

        @fp.fpy(ctx=fp.REAL)
        def caller(x):
            return leaf(x)

        mod = fp.Module()
        mod.add(caller)
        names = {f.name for f in Specialize.apply(mod).functions()}
        leaf_name = next(n for n in names if n.startswith('leaf'))
        assert leaf_name.count('__') == 1, leaf_name


class TestParamTermsFollowTheParametersTheyBindTo:
    """A `DigitBoundParams`' terms are bound to parameters by position, so
    dropping a dead parameter has to drop its term too -- otherwise every later
    parameter inherits the one before it, and `zip` hides the mismatch."""

    def test_a_dropped_parameter_takes_its_term(self):
        @fp.fpy(ctx=fp.REAL)
        def helper(unused, x, y):
            with fp.MPFixedContext(-8, fp.RM.RTZ):
                return fp.round(x) + fp.round(y)

        @fp.fpy(ctx=fp.REAL)
        def main(p, q, s):
            return helper(p, q, s)

        mod = _module((main, [RealType(fp.FP64), RealType(fp.FP16),
                              RealType(fp.FP64)]))
        bound_params: dict = {}
        specs = {
            f.name: f
            for f in Specialize.apply(mod, bound_params=bound_params).functions()
        }

        helpers = [n for n in specs if n.startswith('helper')]
        assert len(helpers) == 1
        # `unused` is dead and goes, so the spec is down to two parameters
        assert len(specs[helpers[0]].ast.args) == 2
        for name, params in bound_params.items():
            assert len(params.args) == len(specs[name].ast.args), (
                f'{name}: {len(params.args)} terms for '
                f'{len(specs[name].ast.args)} parameters'
            )


class TestCallersThatBoundACalleeDifferentlyDoNotShare:
    """A spec is analyzed once, with one caller's params.  Two callers that
    derive different bounds inside the same callee must therefore take
    separate specs, or whichever was reached first decides the other's
    storage."""

    @staticmethod
    def _rnd_specs(fn):
        mod = _module((fn, [RealType(fp.FP64), RealType(fp.FP64)]))
        out = Specialize.apply(mod)
        return sorted(f.name for f in out.functions() if f.name.startswith('rnd'))

    def test_a_tied_and_an_untied_position_are_two_specs(self):
        @fp.fpy(ctx=fp.REAL)
        def rnd(x, n):
            with fp.MPFixedContext(n, fp.RM.RTZ):
                return fp.round(x)      # no assignment: `by_def` is the params

        @fp.fpy(ctx=fp.REAL)
        def tied_first(a, b):
            u = rnd(a, fp.logb(a) - 12)     # position tied to the value rounded
            v = rnd(a, fp.logb(b) - 12)     # position unrelated to it
            return u + v

        @fp.fpy(ctx=fp.REAL)
        def untied_first(a, b):
            v = rnd(a, fp.logb(b) - 12)
            u = rnd(a, fp.logb(a) - 12)
            return u + v

        first = self._rnd_specs(tied_first)
        assert len(first) == 2, first
        # ... and which caller came first must not decide the outcome
        assert first == self._rnd_specs(untied_first)

    def test_the_same_bounds_on_different_expressions_are_two_specs(self):
        """Each call ties one rounding tight and leaves the other loose, so
        the formats match as a multiset but not expression by expression; one
        spec would bound one call by the other's terms."""
        @fp.fpy(ctx=fp.REAL)
        def rnd(x, n):
            with fp.MPFixedContext(n, fp.RM.RTZ):
                return fp.round(x)

        @fp.fpy(ctx=fp.REAL)
        def both(x, y, p, q):
            return rnd(x, p) + rnd(y, q)

        @fp.fpy(ctx=fp.REAL)
        def f(a, b, c):
            s = both(a, b, fp.logb(a) - 3, fp.logb(c) - 3)
            t = both(a, b, fp.logb(c) - 3, fp.logb(b) - 3)
            return s + t

        mod = _module((f, [RealType(fp.FP32)] * 3))
        out = Specialize.apply(mod, size_key=True, bound_params={})
        names = [g.name for g in out.functions() if g.name.startswith('both')]
        assert len(names) == 2, names


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
