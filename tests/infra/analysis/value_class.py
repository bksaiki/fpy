"""
Value-class analysis integration tests.

The claims are also *checked against a run*: every expression is probed as the
program executes and its value compared with the class the analysis gave it.
The analysis is sound only if no observed value falls outside its class, and
that is the property nothing else here tests -- the unit sweeps cover the
transfer functions, not the flow through a whole program.
"""

import signal
from fractions import Fraction

import fpy2 as fp
from fpy2.analysis import ValueClass, ValueClassInfer, class_of
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

_SECOND_ARG = [float('nan'), float('inf'), 0.0, 2.5]
"""Arguments after the first are sampled rather than crossed, which would raise
the grid to a power for no new classes."""

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
        second = _LIST_VALUES if kinds[1] else _SECOND_ARG
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

_MIN_INFORMATIVE = 1000
"""A claim of the top class cannot be contradicted, so a run comparing only
against it proves nothing.  The corpus gives six figures of informative
comparisons across 114 of its functions; this is a floor well under that, to
catch the check going vacuous rather than to pin a number."""

_SECONDS = 2.0
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


def _observed_class(v) -> 'ValueClass | None':
    """The class of a runtime value, or ``None`` where it carries none."""
    match v:
        case Float():
            return class_of(v)
        case Fraction():
            return class_of(Float.from_rational(v, ctx=REAL))
        case _:
            return None


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


def _check_against_a_run(func: fp.Function) -> 'tuple[list[str], int]':
    """``(contradictions, informative comparisons)`` from running *func* over
    the probe values, in each of :func:`_forms`.

    Driven under ``REAL``, which is the only context where the analysis says
    anything: under a concrete one :meth:`_rounded` reports the classes that
    context can represent, and for ``FP64`` that is every class -- a claim no
    run can contradict.  The second return value counts the comparisons that
    were *not* against the top, so a check that has quietly gone vacuous is
    visible rather than green.
    """
    kinds = _arg_kinds(func)
    if kinds is None:
        return [], 0
    bad: list[str] = []
    informative = 0
    for form in _forms(func):
        found, n = _check_one_form(form, kinds)
        bad += found
        informative += n
    return bad, informative


def _check_one_form(
    func: fp.Function, kinds: 'list[bool]'
) -> 'tuple[list[str], int]':
    info = ValueClassInfer.analyze(func.ast)
    bad: list[str] = []
    informative = 0
    compiler: BytecodeCompiler

    def probe(i: int, v):
        nonlocal informative
        e = compiler.probed[i]
        claimed = info.by_expr.get(e)
        seen = _observed_class(v)
        if not isinstance(claimed, ValueClass) or seen is None:
            return v
        if claimed != ValueClass.TOP:
            informative += 1
        if not (seen & claimed):
            bad.append(f'{func.name}: `{e.format()}` is {seen}, claimed {claimed}')
        return v

    compiler = BytecodeCompiler(func.ast, func.env, probe=probe)
    fn = compiler.compile()

    grid = _grid(kinds)

    def run_grid():
        for args in grid:
            try:
                fn(*[to_value(a) for a in args], REAL)
            except _Timeout:
                raise
            except Exception:  # noqa: BLE001, S112 -- a refusal has no class
                continue

    _with_timeout(_SECONDS, run_grid)
    return bad, informative


def _test_value_class_against_runs():
    bad: list[str] = []
    informative = 0
    for core in all_tests():
        if core.name in _unit_ignore:
            continue
        try:
            found, n = _check_against_a_run(core)
        except Exception as exc:   # noqa: BLE001 -- not every example is drivable
            print(f'  {core.name}: not driven ({type(exc).__name__})')
            continue
        bad += found
        informative += n
    print(f'value classes checked against runs: {informative} informative')
    assert not bad, 'value class contradicted by a run:\n  ' + '\n  '.join(bad)
    assert informative >= _MIN_INFORMATIVE, (
        f'only {informative} comparisons were against anything but the top '
        f'class; the check has gone vacuous'
    )


def test_value_class():
    _test_value_class_unit()
    _test_value_class_library()
    _test_value_class_against_runs()


if __name__ == '__main__':
    test_value_class()
