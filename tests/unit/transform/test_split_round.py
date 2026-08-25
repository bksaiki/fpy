"""
Unit tests for the :class:`fpy2.transform.SplitRound` transform.

The candidates are *rounded* operations — the complement of
:class:`fpy2.transform.RoundInsert`'s — and the rewrite mints fresh ``_tN``
names, so these tests assert

1. **Structural shape**: the operation lands alone in a block of the
   intermediate, with an explicit ``round`` back to the target in the enclosing
   block, which is what applies the target's own mode.
2. **Semantic equivalence** via the interpreter: a split admitted by Figure 8
   changes no result, checked over random inputs and the edge cases the
   premises exist for (underflow, overflow, ``-0``).
3. **Declines**, each by the reason it gives.
4. **The `where` contract**: a listing reports exactly what ``where=None``
   rewrites.
"""

import pathlib
import random

import pytest

import fpy2 as fp
from fpy2.analysis.format_infer import derive_intermediate
from fpy2.analysis.format_infer import AbstractFormat
from fpy2.ast.fpyast import ContextStmt, ForeignVal, FuncDef, Mul, Round
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import RoundingMode as RM
from fpy2.transform import (
    ExprCursor,
    Monomorphize,
    SplitRound,
    TransformDeclined,
    TransformReferenceError,
)
from fpy2.transform.cursor import expr_sites
from fpy2.types import RealType

VIA32 = derive_intermediate(fp.FP32)
"""The tightest RTO intermediate for an FP32 / RNE target."""


def _count(ast: FuncDef, kind) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, c):
            nonlocal n
            if kind is ContextStmt:
                n += 1
            super()._visit_context(stmt, c)

        def _visit_unaryop(self, e, c):
            nonlocal n
            if kind is Round and isinstance(e, Round):
                n += 1
            super()._visit_unaryop(e, c)

    _C()._visit_function(ast, None)
    return n


def _via_blocks(ast: FuncDef, ctx) -> int:
    """Blocks in *ast* written as a `ForeignVal` of *ctx*."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, c):
            nonlocal n
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val == ctx:
                n += 1
            super()._visit_context(stmt, c)

    _C()._visit_function(ast, None)
    return n


def _load(src: str, tag: str):
    """An FPy function from generated source, since `fp.fpy` needs a real file."""
    import importlib.util
    import sys
    import tempfile

    d = pathlib.Path(tempfile.gettempdir()) / 'fpy_split_probes'
    d.mkdir(exist_ok=True)
    path = d / f'probe_{abs(hash(tag + src))}.py'
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.probe


def _tight_via(rm, overflow=None):
    """An intermediate bounded one step past FP32's range -- the width where the
    boundary is observable."""
    f32 = AbstractFormat.from_format(fp.FP32.format())
    fmt = f32.next_bound().with_prec_offset(1).with_exp_offset(-1).format()
    args = (fmt.pmax, fmt.emin, fmt.pos_maxval, rm)
    if overflow is not None:
        return fp.MPBFloatContext(*args, overflow, neg_maxval=fmt.neg_maxval)
    return fp.MPBFloatContext(*args, neg_maxval=fmt.neg_maxval)


def _cursor_at(ast: FuncDef, kind) -> ExprCursor:
    found = expr_sites(ast, lambda e: isinstance(e, kind))
    assert found, f'no {kind.__name__} in the program'
    return found[0]


def _agree(before: FuncDef, after: FuncDef, runtime, args_list) -> bool:
    fa: Function = Function(before, runtime=runtime)
    fb: Function = Function(after, runtime=runtime)
    return all(str(fa(*a)) == str(fb(*a)) for a in args_list)


@fp.fpy(ctx=fp.REAL)
def _product(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        t = x * y
    return t


@fp.fpy(ctx=fp.REAL)
def _two_ops(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        t = x * y
        s = x + y
    return t + s


@fp.fpy(ctx=fp.REAL)
def _exact(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.REAL:
        t = x * y
    return t


class TestShape:
    def test_the_operation_moves_under_the_intermediate(self):
        out = SplitRound.apply(_product.ast, VIA32)
        assert _via_blocks(out, VIA32) == 1
        # the re-rounding is explicit, since an assignment rounds nothing
        assert _count(out, Round) == 1

    def test_the_round_sits_outside_the_block(self):
        """The `round` has to be in the *enclosing* block to pick up the
        target's mode; inside the intermediate's block it would be a no-op."""
        out = SplitRound.apply(_product.ast, VIA32)
        text = Function(out, runtime=_product.runtime).format()
        via_line = next(i for i, ln in enumerate(text.splitlines()) if 'RTO' in ln)
        round_line = next(i for i, ln in enumerate(text.splitlines()) if 'fp.round' in ln)
        assert round_line > via_line
        # and at a shallower indent than the operation it rounds
        lines = text.splitlines()
        op_indent = len(lines[via_line + 1]) - len(lines[via_line + 1].lstrip())
        rd_indent = len(lines[round_line]) - len(lines[round_line].lstrip())
        assert rd_indent < op_indent

    def test_each_site_gets_its_own_block(self):
        out = SplitRound.apply(_two_ops.ast, VIA32)
        assert _via_blocks(out, VIA32) == 2
        assert _count(out, Round) == 2


class TestEquivalence:
    @staticmethod
    def _sweep(n: int = 2000):
        rng = random.Random(0)
        for i in range(n):
            a = rng.uniform(-1e3, 1e3) if i % 3 else rng.uniform(-1e-30, 1e-30)
            b = rng.uniform(-1e3, 1e3) if i % 2 else rng.uniform(-1e30, 1e30)
            yield a, b

    @staticmethod
    def _subnormal_sweep(n: int = 600):
        """Products landing in FP32's *gradual*-underflow band.

        The `exp - k` half of each premise exists for exactly this range, and
        nothing else here reaches it: the products in `_sweep` either flush to
        zero or stay normal.
        """
        rng = random.Random(1)
        for _ in range(n):
            k = rng.randint(-74, -60)
            yield rng.uniform(1.0, 2.0) * 2.0 ** k, rng.uniform(0.5, 2.0)

    @staticmethod
    def _overflow_sweep(n: int = 400):
        """Products above FP32's maxval, where a non-saturating intermediate
        sends to `inf` what the target clamps."""
        rng = random.Random(2)
        for _ in range(n):
            yield rng.uniform(1.0, 2.0) * 2.0 ** 127, rng.uniform(2.0, 8.0)

    def test_a_split_changes_no_value(self):
        out = SplitRound.apply(_product.ast, VIA32)
        assert _agree(_product.ast, out, _product.runtime, self._sweep())

    def test_the_edge_cases_the_premises_exist_for(self):
        """Underflow to zero, overflow to infinity, and `-0` — the cases where a
        careless intermediate would disagree."""
        out = SplitRound.apply(_product.ast, VIA32)
        edges = [
            (1e-40, 1e-40), (1e30, 1e30), (-1e30, 1e30),
            (0.0, -1.0), (-0.0, 1.0), (1.0, 1.0),
            (float('nan'), 1.0), (float('inf'), 0.0),
        ]
        assert _agree(_product.ast, out, _product.runtime, edges)

    def test_gradual_underflow(self):
        out = SplitRound.apply(_product.ast, VIA32)
        assert _agree(_product.ast, out, _product.runtime, self._subnormal_sweep())

    def test_overflow_matches_the_unsplit_program(self):
        """The derived intermediate is unbounded, so the only rounding that can
        overflow is the target's -- as in the program before the split.  A
        *bounded* intermediate has an overflow of its own that no single
        behaviour gets right: a clamping target (RTZ) needs it not to overflow,
        an overflowing one (RTO) needs it to."""
        out = SplitRound.apply(_product.ast, VIA32)
        assert _agree(_product.ast, out, _product.runtime, self._overflow_sweep())

    @pytest.mark.parametrize('rm1', [RM.RTZ, RM.RTO, RM.RNE])
    def test_overflow_for_a_clamping_and_an_overflowing_target(self, rm1):
        """The two directions that a bounded intermediate cannot satisfy at
        once, both exact here."""
        target = fp.FP32.with_params(rm=rm1)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with target:
                t = x * y
            return t

        out = SplitRound.apply(f.ast, derive_intermediate(target))
        assert _agree(f.ast, out, f.runtime, self._overflow_sweep(150))

    def test_a_nested_rounded_operand(self):
        """An operand that is itself a rounded operation has to be bound under
        the *original* scope; left inline it would be re-rounded to the
        intermediate instead of to the target."""

        @fp.fpy(ctx=fp.FP32)
        def nested(x: fp.Real, y: fp.Real) -> fp.Real:
            return (x * y) + y

        assert len(SplitRound.sites(nested.ast, ctx=VIA32)) == 2
        out = SplitRound.apply(nested.ast, VIA32)
        assert _agree(nested.ast, out, nested.runtime, self._sweep(800))
        assert _agree(nested.ast, out, nested.runtime, self._subnormal_sweep(200))

    def test_a_fixed_point_target(self):
        """The premises are containment checks on `A`, indifferent to the format
        family."""
        target = fp.MPFixedContext(-8, fp.RoundingMode.RTZ)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with target:
                t = x * y
            return t

        via = derive_intermediate(target)
        assert len(SplitRound.sites(f.ast, ctx=via)) == 1
        out = SplitRound.apply(f.ast, via)
        assert _agree(f.ast, out, f.runtime, self._sweep(400))

    def test_both_sites_together(self):
        out = SplitRound.apply(_two_ops.ast, VIA32)
        assert _agree(_two_ops.ast, out, _two_ops.runtime, self._sweep(500))

    @pytest.mark.parametrize('rm1', [RM.RTZ, RM.RAZ, RM.RTO])
    def test_a_directed_target(self, rm1):
        """Each admitted final mode, through its own derived intermediate."""
        target = fp.FP32.with_params(rm=rm1)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with target:
                t = x * y
            return t

        via = derive_intermediate(target)
        assert len(SplitRound.sites(f.ast, ctx=via)) == 1
        out = SplitRound.apply(f.ast, via)
        assert _agree(f.ast, out, f.runtime, self._sweep(500))


class TestDeclines:
    def test_an_exact_operation(self):
        """Nothing to split — that direction is `insert_round`'s."""
        assert SplitRound.sites(_exact.ast, ctx=VIA32) == []
        why = SplitRound.refusals(_exact.ast, ctx=VIA32)
        assert len(why) == 1 and 'no rounding to split' in why[0][1]

    def test_an_unsound_mode_pair(self):
        """RNE over RNE: the pairing every `fp.FP*` context falls into, and
        Table 2's last row says it is unsound however wide the intermediate."""
        assert SplitRound.sites(_product.ast, ctx=fp.FP64) == []
        why = SplitRound.refusals(_product.ast, ctx=fp.FP64)
        assert len(why) == 1 and 'is not the same as' in why[0][1]

    def test_a_stochastic_intermediate(self):
        via = VIA32.with_params(num_randbits=2)
        assert via.is_stochastic()
        assert SplitRound.sites(_product.ast, ctx=via) == []
        why = SplitRound.refusals(_product.ast, ctx=via)
        assert len(why) == 1 and 'stochastic' in why[0][1]

    def test_a_stochastic_target(self):
        """The other side of the same condition: the *operation's* scope rounds
        stochastically."""
        target = fp.FP32.with_params(num_randbits=2)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with target:
                t = x * y
            return t

        assert SplitRound.sites(f.ast, ctx=VIA32) == []
        why = SplitRound.refusals(f.ast, ctx=VIA32)
        assert len(why) == 1 and 'stochastic' in why[0][1]

    def test_a_symbolic_scope(self):
        """Without a pinned function context the scope stays symbolic, so the
        rounding it performs is unknown."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y

        assert SplitRound.sites(f.ast, ctx=VIA32) == []
        why = SplitRound.refusals(f.ast, ctx=VIA32)
        assert why and 'symbolic' in why[0][1]

    def test_naming_a_refused_operation_by_cursor(self):
        cursor = _cursor_at(_exact.ast, Mul)
        with pytest.raises(TransformDeclined, match='no rounding to split'):
            SplitRound.apply(_exact.ast, VIA32, where=cursor)

    def test_a_where_naming_nothing(self):
        with pytest.raises(TransformReferenceError):
            SplitRound.apply(_product.ast, VIA32, where=7)

    def test_a_value_past_the_range_the_two_disagree_on(self):
        """A target that *clamps* rather than reaching infinity, over an
        intermediate that overflows: just past the intermediate's bound the
        composition gives infinity where the single rounding gives maxval."""
        via = _tight_via(fp.RoundingMode.RTO, fp.OverflowMode.OVERFLOW)

        @fp.fpy(ctx=fp.FP32.with_params(rm=RM.RTZ))
        def clamps(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y

        assert SplitRound.sites(clamps.ast, ctx=via) == []
        why = SplitRound.refusals(clamps.ast, ctx=via)
        assert len(why) == 1 and 'past the intermediate' in why[0][1]

    def test_the_same_intermediate_saturating_is_admitted(self):
        """The mirror of the above: saturating, the intermediate hands back its
        own maxval, which the target then clamps exactly as it would have."""
        via = _tight_via(fp.RoundingMode.RTO, fp.OverflowMode.SATURATE)

        @fp.fpy(ctx=fp.FP32.with_params(rm=RM.RTZ))
        def clamps(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y

        assert len(SplitRound.sites(clamps.ast, ctx=via)) == 1

    def test_an_intermediate_that_raises_on_the_probe(self):
        """A probe a context cannot answer at all: `OverflowMode.ASSERT` makes
        rounding past the bound an exception, so the rewrite declines rather
        than propagating the error."""
        via = _tight_via(fp.RoundingMode.RTO, fp.OverflowMode.ASSERT)

        @fp.fpy(ctx=fp.FP32.with_params(rm=RM.RTZ))
        def clamps(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y

        assert SplitRound.sites(clamps.ast, ctx=via) == []
        why = SplitRound.refusals(clamps.ast, ctx=via)
        assert len(why) == 1 and 'past the intermediate' in why[0][1]

    def test_a_probe_no_context_here_represents(self):
        """The other refusal: neither format has NaN, so the NaN probe raises.
        Containment still holds — it is the target's specials that must be in
        the intermediate, and it has none."""
        via = fp.MPBFixedContext(
            -2, fp.RealFloat(m=511, exp=-1), RM.RTZ, fp.OverflowMode.SATURATE,
            neg_maxval=fp.RealFloat(m=-512, exp=-1),
        )

        @fp.fpy(ctx=fp.SINT8)
        def product(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y

        assert SplitRound.sites(product.ast, ctx=via) == []
        why = SplitRound.refusals(product.ast, ctx=via)
        assert len(why) == 1 and 'a special' in why[0][1]

    def test_a_non_context_intermediate(self):
        with pytest.raises(TypeError):
            SplitRound.sites(_product.ast, ctx=fp.FP32.format())  # type: ignore[arg-type]


class TestUnreachablePositions:
    def test_a_while_condition_is_left_alone(self):
        """The condition is re-evaluated every iteration, so a block hoisted
        before the loop computes it once — which does not terminate."""

        @fp.fpy(ctx=fp.FP32)
        def shrink(n: fp.Real) -> fp.Real:
            i = 0.0
            while i * 1.0 < n:
                i = i + 1.0
            return i

        out = SplitRound.apply(shrink.ast, VIA32)
        # the body's `i + 1.0` is the one site; the condition's `i * 1.0` is not
        assert len(SplitRound.sites(shrink.ast, ctx=VIA32)) == 1
        assert str(Function(out, runtime=shrink.runtime)(3.0)) == str(shrink(3.0))

    @pytest.mark.parametrize('build', ['if_cond', 'for_iterable', 'comprehension'])
    def test_the_other_positions_with_no_statement_slot(self, build):
        """Each is refused for the same reason as the `while` condition: there
        is nowhere to put the block."""
        if build == 'if_cond':
            @fp.fpy(ctx=fp.FP32)
            def f(x: fp.Real, y: fp.Real) -> fp.Real:
                a = 0.0
                if x * y > 0.0:
                    a = 1.0
                return a
        elif build == 'for_iterable':
            @fp.fpy(ctx=fp.FP32)
            def f(xs: list[fp.Real], y: fp.Real) -> fp.Real:
                a = 0.0
                for v in xs[0:y * 2]:
                    a = v
                return a
        else:
            @fp.fpy(ctx=fp.FP32)
            def f(xs: list[fp.Real], y: fp.Real) -> fp.Real:
                return [v * y for v in xs][0]

        assert SplitRound.sites(f.ast, ctx=VIA32) == []
        why = SplitRound.refusals(f.ast, ctx=VIA32)
        assert why and all('no statement-level position' in r for _, r in why)

    def test_an_if_expr_branch_is_left_alone(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, c: bool) -> fp.Real:
            return (x * x) if c else x

        assert SplitRound.sites(f.ast, ctx=VIA32) == []


class TestWhereContract:
    def test_sites_are_what_where_none_rewrites(self):
        listed = SplitRound.sites(_two_ops.ast, ctx=VIA32)
        assert len(listed) == 2
        every = SplitRound.apply(_two_ops.ast, VIA32)
        assert _via_blocks(every, VIA32) == len(listed)

    def test_each_index_rewrites_one(self):
        for j in range(len(SplitRound.sites(_two_ops.ast, ctx=VIA32))):
            out = SplitRound.apply(_two_ops.ast, VIA32, where=j)
            assert _via_blocks(out, VIA32) == 1

    def test_a_cursor_aims_the_same_as_its_index(self):
        listed = SplitRound.sites(_two_ops.ast, ctx=VIA32)
        for j, cursor in enumerate(listed):
            by_index = SplitRound.apply(_two_ops.ast, VIA32, where=j)
            by_cursor = SplitRound.apply(_two_ops.ast, VIA32, where=cursor)
            assert by_cursor.is_equiv(by_index)


class TestBoundedIntermediate:
    """A bounded intermediate is safe exactly where the operation cannot reach
    its range, which format inference can prove from the argument formats."""

    RTO64 = fp.FP64.with_params(rm=fp.RoundingMode.RTO)

    @staticmethod
    def _pinned(target=fp.FP32):
        """`x * y` at *target* with FP32 arguments, so the exact product has a
        bound: 48 digits, magnitude under FP32's maxval squared."""

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with target:
                t = x * y
            return t

        return Monomorphize.apply(
            f.ast, fp.REAL, [RealType(fp.FP32), RealType(fp.FP32)],
        )

    def test_a_provably_unreachable_range_is_admitted(self):
        """FP32 x FP32 cannot leave FP64's range, so the intermediate never
        overflows and the composition is the finite-value case the theorems
        cover."""
        ast = self._pinned()
        assert len(SplitRound.sites(ast, ctx=self.RTO64)) == 1

    @pytest.mark.parametrize('rm1', [RM.RNE, RM.RTZ, RM.RTO])
    def test_and_it_is_exact(self, rm1):
        """Including at the target's overflow boundary, which is where a tight
        bounded intermediate goes wrong."""
        ast = self._pinned(fp.FP32.with_params(rm=rm1))
        out = SplitRound.apply(ast, self.RTO64)
        edges = [(1.0e38, 4.0), (float(2 ** 127), 2.0), (3.4e38, 1.0000001),
                 (1.7014118346046923e+38, 1.9999999), (1e-40, 1e-40), (1.0, 1.0)]
        assert _agree(ast, out, None, edges)

    def test_a_range_the_operation_could_exceed_is_declined(self):
        """An intermediate bounded just above the target, overflowing where the
        target clamps: neither the range proof nor the boundary check holds."""
        via = _tight_via(fp.RoundingMode.RTO, fp.OverflowMode.OVERFLOW)
        ast = self._pinned(fp.FP32.with_params(rm=RM.RTZ))
        assert SplitRound.sites(ast, ctx=via) == []

    def test_a_target_that_overflows_throughout_the_range_is_admitted(self):
        """The boundary is unobservable when everything past the intermediate's
        range is already infinite to the target, so a value it rounds finitely
        and one it overflows both end up infinite."""
        via = fp.FP32.with_params(rm=fp.RoundingMode.RTO)

        @fp.fpy(ctx=fp.FP16)
        def narrow(x: fp.Real) -> fp.Real:
            return x + x

        assert len(SplitRound.sites(narrow.ast, ctx=via)) == 1
        out = SplitRound.apply(narrow.ast, via)
        edges = [(1.0,), (32752.0,), (65504.0,), (1e38,), (3.4e38,),
                 (1e-8,), (0.0,), (-65504.0,)]
        assert _agree(narrow.ast, out, narrow.runtime, edges)

    def test_an_unbounded_intermediate_needs_no_proof(self):
        """It cannot overflow, so the argument formats are irrelevant."""
        assert len(SplitRound.sites(_product.ast, ctx=VIA32)) == 1


class TestNotCandidates:
    @pytest.mark.parametrize('op', ['round', 'cast'])
    def test_an_explicit_rounding_is_not_split(self, op):
        """Splitting a rounding is the inverse rewrite, and admitting these
        would make a second application grow the tree twice as fast."""

        if op == 'round':
            @fp.fpy(ctx=fp.REAL)
            def f(x: fp.Real) -> fp.Real:
                with fp.FP32:
                    t = fp.round(x)
                return t
        else:
            @fp.fpy(ctx=fp.REAL)
            def f(x: fp.Real) -> fp.Real:
                with fp.FP32:
                    t = fp.cast(x)
                return t

        assert SplitRound.sites(f.ast, ctx=VIA32) == []
        assert SplitRound.refusals(f.ast, ctx=VIA32) == []


class TestRepeatedApplication:
    def test_each_pass_terminates_and_splits_once_more(self):
        """Like the loop rewrites, this one is not idempotent: the operation is
        now under an RTO intermediate, and RTO-over-RTO is itself admitted.  One
        pass terminates; a schedule that wants a fixpoint has to bound it."""
        ast = _product.ast
        for expect in (1, 2, 3):
            assert len(SplitRound.sites(ast, ctx=VIA32)) == 1
            ast = SplitRound.apply(ast, VIA32)
            assert _via_blocks(ast, VIA32) == expect
            assert str(Function(ast, runtime=_product.runtime)(1.5, 2.5)) \
                == str(_product(1.5, 2.5))


class TestAnyRealValuedOperation:
    """The rules quantify over an arbitrary real, so what produced it does not
    matter: every operation that rounds its result to the active context is
    splittable, not just the arithmetic ones."""

    ROUNDS = {
        'sqrt': 'fp.sqrt(x)', 'div': 'x / y', 'fma': 'fp.fma(x, y, x)',
        'sin': 'fp.sin(x)', 'exp': 'fp.exp(x)', 'hypot': 'fp.hypot(x, y)',
        'floor': 'fp.floor(x)', 'pow': 'x ** y', 'atan2': 'fp.atan2(x, y)',
        'fmod': 'fp.fmod(x, y)', 'cbrt': 'fp.cbrt(x)',
    }

    @pytest.mark.parametrize('name', sorted(ROUNDS))
    def test_it_splits_and_agrees(self, name):
        src = (
            'import fpy2 as fp\n\n'
            '@fp.fpy(ctx=fp.FP32)\n'
            'def probe(x: fp.Real, y: fp.Real) -> fp.Real:\n'
            f'    return {self.ROUNDS[name]}\n'
        )
        probe = _load(src, name)
        assert len(SplitRound.sites(probe.ast, ctx=VIA32)) == 1
        out = SplitRound.apply(probe.ast, VIA32)
        args = [(1.1, 1.3), (2.0, 0.5), (7.25, 3.5), (0.5, 4.0)]
        assert _agree(probe.ast, out, probe.runtime, args)

    def test_an_irrational_constant_splits(self):
        """`pi` is a real rounded to the context like any other."""

        @fp.fpy(ctx=fp.FP32)
        def f() -> fp.Real:
            return fp.const_pi()

        assert len(SplitRound.sites(f.ast, ctx=VIA32)) == 1
        out = SplitRound.apply(f.ast, VIA32)
        assert _agree(f.ast, out, f.runtime, [()])

    @pytest.mark.parametrize('expr', ['min(x, y)', 'max(x, y)'])
    def test_min_and_max_are_not_sites(self, expr):
        """They *select* an argument and hand it back with its own format
        rather than rounding to the active context, so there is no rounding to
        split.  Measured: splitting `min` disagreed on 146 of 154 inputs."""
        src = (
            'import fpy2 as fp\n\n'
            '@fp.fpy(ctx=fp.FP32)\n'
            'def probe(x: fp.Real, y: fp.Real) -> fp.Real:\n'
            f'    return {expr}\n'
        )
        probe = _load(src, expr[:3])
        assert SplitRound.sites(probe.ast, ctx=VIA32) == []

    @pytest.mark.parametrize('expr,ret', [
        ('len(zs)', 'fp.Real'), ('fp.isnan(x)', 'bool'), ('fp.inf()', 'fp.Real'),
    ])
    def test_the_non_rounding_operations_are_not_sites(self, expr, ret):
        """An exact query, a boolean and a non-finite constant: none is a real
        computation followed by a rounding."""
        src = (
            'import fpy2 as fp\n\n'
            '@fp.fpy(ctx=fp.FP32)\n'
            f'def probe(x: fp.Real, zs: list[fp.Real]) -> {ret}:\n'
            f'    return {expr}\n'
        )
        probe = _load(src, 'nr')
        assert SplitRound.sites(probe.ast, ctx=VIA32) == []
