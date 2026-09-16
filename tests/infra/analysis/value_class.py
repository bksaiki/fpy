"""
Value-class analysis integration tests.

The claims are also *checked against a run*: the interpreter's hook sees every
expression as the program executes, and its value is compared with the class the
analysis gave it.
The analysis is sound only if no observed value falls outside its class, and
that is the property nothing else here tests -- the unit sweeps cover the
transfer functions, not the flow through a whole program.
"""

import signal
from fractions import Fraction

import fpy2 as fp
from fpy2.analysis import ValueClass, ValueClassInfer, class_of
from fpy2.analysis.value_class import (
    ClassBound,
    ListClass,
    TupleClass,
    ValueClassAnalysis,
    join_class,
)
from fpy2.ast.fpyast import Var
from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.interpret.byte import BytecodeCompiler
from fpy2.interpret.value import to_value
from fpy2.number import REAL, Float
from fpy2.types import RealType

from ..examples import all_tests

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]

_unit_ignore: list[str] = [
    # These examples use dynamically constructed or parametric contexts that
    # cannot be resolved statically by TypeInfer or ContextUse.
    'test_context_expr1',
    'test_context_expr2',
    'test_context8',
    'example_static_context1',
    'example_static_context2',
    'keep_p_1'
]


def _test_value_class_unit():
    for core in all_tests():
        assert isinstance(core, fp.Function)
        if core.name in _unit_ignore:
            continue

        print(core.name)
        info = ValueClassInfer.analyze(core.ast)
        for d, cls in info.by_def.items():
            print(f'  {d.name} -> {cls}')


def _test_value_class_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            match obj:
                case fp.Function():
                    print(obj.name)
                    info = ValueClassInfer.analyze(obj.ast)
                    for d, cls in info.by_def.items():
                        print(f'  {d.name} -> {cls}')


_PROBE_VALUES = [
    float('nan'), float('inf'), float('-inf'), 0.0, -0.0,
    1.0, -1.0, 0.5, -2.5, 3.0, 1e300, 1e-300,
]
"""One per atom, and a few finites: the point is to reach every class."""

_SECOND_ARG = [float('nan'), float('inf'), float('-inf'), 0.0, 2.5, 1e300]
"""Arguments after the first are sampled rather than crossed, which would raise
the grid to a power for no new classes.  ``1e300`` is not a class of its own --
it is there so a *product* overflows, which is the only way a concrete context
turns a finite operand into an infinity."""

_LIST_VALUES = [
    [1.0, -2.5, 3.0],
    [1.0, 0.0, 3.0],
    [1.0, float('nan'), 3.0],
    [1.0, float('-inf'), float('inf')],
    [],
]
"""One list per class an *element* can be, plus an empty one -- what a fact
about a whole list has to be checked against.  The first is the filler."""


def _grid(kinds: 'list[bool]') -> list[tuple]:
    """Argument tuples reaching every class in every position.  ``kinds[i]`` is
    true for a list argument, whose samples vary an *element*'s class."""
    cols = [_LIST_VALUES if k else _PROBE_VALUES for k in kinds]
    n = len(kinds)
    if n == 0:
        return [()]
    if n == 1:
        return [(v,) for v in cols[0]]
    if n == 2:
        second = cols[1] if kinds[1] else _SECOND_ARG
        return [(x, y) for x in cols[0] for y in second]
    # beyond two, every position at once and then one at a time
    out = [
        tuple(col[j % len(col)] for col in cols)
        for j in range(len(_PROBE_VALUES))
    ]
    filler = [_LIST_VALUES[0] if k else 1.0 for k in kinds]
    for i in range(n):
        for v in cols[i]:
            args = list(filler)
            args[i] = v
            out.append(tuple(args))
    return out

_MIN_INFORMATIVE_EXPRS = 850
_MIN_INFORMATIVE_FUNCS = 115
_MIN_LOWERED = 150
_MIN_INFORMATIVE_LISTS = 1
"""Floors, to catch the check going quiet rather than to pin a number.

A claim of the top class cannot be contradicted, so a run comparing only
against it proves nothing.  Counted as *distinct expressions* rather than
comparisons: a comparison count is dominated by whichever program loops longest
before :data:`_SECONDS` interrupts it, which makes it a measure of how fast the
machine is.  The per-function floor is what catches one shape going dark, and
:data:`_MIN_LOWERED` catches :func:`_forms` silently failing to lower -- the
lowered form is the only one with an element store or a reduction loop in it.
"""

_CONTEXTS = (REAL, fp.FP64)
"""Runtime contexts each program is driven under.

``REAL`` is where an exact claim is testable at all.  A concrete one is where a
program with a *symbolic* context rounds -- and rounding is what turns a finite
value into an infinity, which an exact-only run never sees.
"""

_SECONDS = 4.0
"""Wall clock per function.  An ``inf`` reaching a loop bound runs forever, so
the grid is interrupted rather than bounded -- the same reason
``tests.infra.backend.cpp`` carries a timeout."""


class _Timeout(Exception):
    pass


def _with_timeout(seconds: float, run):
    def _handler(signum, frame):
        raise _Timeout

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        run()
    except _Timeout:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _observed_class(v) -> ClassBound:
    """A runtime value's class, shaped like the value.

    Structural, because a claim about a list is: ``by_elt`` and ``bound_of``
    are what a storage consumer reads, and a scalar-only observation compares
    against neither.  A list contributes the join over its elements, and an
    empty one contributes bottom, which contradicts nothing.
    """
    match v:
        case Float():
            return class_of(v)
        case Fraction():
            return class_of(Float.from_rational(v, ctx=REAL))
        case list():
            elt: ClassBound = ValueClass(0)
            for x in v:
                elt = join_class(elt, _observed_class(x))
            return ListClass(elt)
        case tuple():
            return TupleClass(tuple(_observed_class(x) for x in v))
        case _:
            return None


def _claimed_class(info: 'ValueClassAnalysis', e) -> ClassBound:
    """What the analysis says *e* is, shaped like the value.

    ``by_expr`` answers for a scalar; a list carries its class per *definition*,
    so a name is followed to the one it reads.
    """
    cls = info.by_expr.get(e)
    if isinstance(cls, ValueClass):
        return cls
    if isinstance(e, Var):
        try:
            return info.bound_of(info.type_info.def_use.find_def_from_use(e))
        except KeyError:
            return None
    return None


def _contradicts(seen: ClassBound, claimed: ClassBound) -> bool:
    """Whether *seen* is a value *claimed* rules out.

    A bottom observation -- an empty list -- rules out nothing, and so does a
    shape the claim does not match.
    """
    match seen, claimed:
        case ValueClass(), ValueClass():
            return bool(seen) and not (seen & claimed)
        case ListClass(), ListClass():
            return _contradicts(seen.elt, claimed.elt)
        case TupleClass(), TupleClass() if len(seen.elts) == len(claimed.elts):
            return any(
                _contradicts(s, c) for s, c in zip(seen.elts, claimed.elts)
            )
        case _:
            return False


def _is_informative(b: ClassBound) -> bool:
    """Whether *b* rules anything out at all."""
    match b:
        case ValueClass():
            return b != ValueClass.TOP
        case ListClass():
            return _is_informative(b.elt)
        case TupleClass():
            return any(_is_informative(x) for x in b.elts)
        case _:
            return False


def _arg_kinds(func: fp.Function) -> 'list[bool] | None':
    """One entry per argument, true for a list of reals -- or ``None`` where
    some argument is neither, the probe values being reals and lists of them."""
    kinds: list[bool] = []
    for a in func.ast.args:
        match a.type:
            case fp.ast.RealTypeAnn():
                kinds.append(False)
            case fp.ast.ListTypeAnn(elt=fp.ast.RealTypeAnn()):
                kinds.append(True)
            case _:
                return None
    return kinds


def _forms(func: fp.Function) -> list[fp.Function]:
    """*func* as written, and as the C++ backend analyzes it.

    A comprehension is an expression and `any` / `all` a single node, so the
    written form has no element store and no reduction loop -- the two shapes a
    class for a list's elements comes from.  Both are checked, since a consumer
    may analyze either.
    """
    out = [func]
    try:
        m = fp.Module()
        m.add(func)
        out += [
            spec for spec in CppCompiler().specialize(m)
            if spec.name == func.name
        ]
    except Exception:  # noqa: BLE001 -- not every example specializes
        pass
    return out


def _check_against_a_run(func: fp.Function) -> 'tuple[list[str], int, int, int]':
    """``(contradictions, informative scalars, informative lists, forms)`` from
    running *func* over the probe values, in each of :func:`_forms`.

    Driven under each of :data:`_CONTEXTS`.  The claims are static, so any
    runtime context may be compared against them -- and a program whose own
    context is symbolic says something different at each, which is what makes
    the concrete one worth the second pass.  The counts are what
    :data:`_MIN_INFORMATIVE_EXPRS` and friends hold, so a check that has
    quietly gone vacuous is visible rather than green.
    """
    kinds = _arg_kinds(func)
    if kinds is None:
        return [], 0, 0, 0
    bad: list[str] = []
    informative = structural = 0
    forms = _forms(func)
    for form in forms:
        found, n, k = _check_one_form(form, kinds)
        bad += found
        informative += n
        structural += k
    return bad, informative, structural, len(forms) - 1


def _check_one_form(
    func: fp.Function, kinds: 'list[bool]'
) -> 'tuple[list[str], int, int]':
    info = ValueClassInfer.analyze(func.ast)
    bad: list[str] = []
    informative: set[int] = set()
    structural: set[int] = set()
    compiler: BytecodeCompiler

    def observe(i: int, v):
        e = compiler.hook_sites[i]
        claimed = _claimed_class(info, e)
        seen = _observed_class(v)
        if claimed is None or seen is None:
            return v
        if _is_informative(claimed):
            (structural if isinstance(claimed, ListClass | TupleClass)
             else informative).add(i)
        if _contradicts(seen, claimed):
            bad.append(f'{func.name}: `{e.format()}` is {seen}, claimed {claimed}')
        return v

    compiler = BytecodeCompiler(func.ast, func.env, hook=observe)
    fn = compiler.compile()

    grid = _grid(kinds)

    def run_grid():
        for ctx in _CONTEXTS:
            for args in grid:
                try:
                    fn(*[to_value(a) for a in args], ctx)
                except _Timeout:
                    raise
                except Exception:  # noqa: BLE001, S112 -- a refusal has no class
                    continue

    _with_timeout(_SECONDS, run_grid)
    return bad, len(informative), len(structural)


def _test_value_class_against_runs():
    bad: list[str] = []
    exprs = funcs = lists = lowered = 0
    for core in all_tests():
        if core.name in _unit_ignore:
            continue
        try:
            found, n, k, forms = _check_against_a_run(core)
        except Exception as exc:   # noqa: BLE001 -- not every example is drivable
            print(f'  {core.name}: not driven ({type(exc).__name__})')
            continue
        bad += found
        exprs += n
        lists += k
        funcs += bool(n)
        lowered += forms
    print(
        f'value classes checked against runs: {exprs} expressions over '
        f'{funcs} functions, {lists} structural, {lowered} lowered forms'
    )
    assert not bad, 'value class contradicted by a run:\n  ' + '\n  '.join(bad)
    assert exprs >= _MIN_INFORMATIVE_EXPRS and funcs >= _MIN_INFORMATIVE_FUNCS, (
        f'only {exprs} expressions over {funcs} functions were compared '
        f'against anything but the top class; the check has gone vacuous'
    )
    assert lowered >= _MIN_LOWERED, (
        f'only {lowered} functions produced a lowered form; the shapes this '
        f'check exists for live there'
    )
    assert lists >= _MIN_INFORMATIVE_LISTS, (
        f'only {lists} list or tuple claims were compared against anything but '
        f'the top class; `by_elt` and `bound_of` are unchecked'
    )


def test_value_class():
    _test_value_class_unit()
    _test_value_class_library()
    _test_value_class_against_runs()


if __name__ == '__main__':
    test_value_class()
