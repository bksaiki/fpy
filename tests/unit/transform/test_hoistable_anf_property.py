"""Property tests for :class:`fpy2.transform.Hoistable`, and :class:`fpy2.transform.ANF`
on its output, over generated programs.

``test_hoistable.py`` and ``test_anf.py`` pin shapes on hand-written programs,
and the profiles measure the corpus; neither goes deep.  The corpus is shallow --
the residue measurement found nothing nested more than two levels -- so the cases
where a lowering meets another lowering are only reachable by generating them.
One test, since generating the program is nearly all of the cost.

For each pass, on the same draw:

1. **It applies.**  ``apply`` runs the syntax checker itself, so this also
   asserts the output is a well-formed program.
2. **Its postcondition.**  For ``Hoistable``, ``refusals`` is empty: a temporary
   may be hoisted out of anywhere in the output, however deeply the lowerings
   nested -- empty outright, since
   :data:`~tests.unit.generators.profiles.ANF_PROFILE` draws no comprehension.
   For ``ANF``, no refusal is in one of the three positions the cpp emitter
   hoists out of (three miscompiles in ``docs/todos/backend-cpp.md``).
3. **Idempotence.**  A second application changes nothing.
4. **Semantics.**  The interpreter agrees before and after, *including on which
   exception it raises* -- the property that catches an ordering regression,
   where a lowering hoisted above a left operand runs the operands out of turn,
   or an eagerly evaluated ternary arm or short-circuited operand raises where
   FPy returns.

``max_depth`` is 3--5, not lower: at depth 2 a ternary's arms are usually
leaves, so it is already in normal form and few draws lower one.

Generation cannot reach a lowering inside a rotated condition -- the generator's
loop template has a pure condition by construction -- so that composition is
covered by hand in ``test_hoistable.py::TestRotation``.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fpy2 as fp
from fpy2.transform import ANF, Hoistable
from fpy2.types import BoolType, RealType

from ..generators.fpy_program import fpy_funcdef, value_for_type
from ..generators.profiles import ANF_PROFILE

_DANGEROUS = (
    'a ternary arm is evaluated conditionally',
    'a short-circuited operand may not be evaluated',
    'a `while` condition is re-evaluated every iteration',
)


@st.composite
def _program(draw):
    """A generated function and one argument vector for it."""
    return_type = draw(st.sampled_from([RealType(), BoolType()]))
    n = draw(st.integers(1, 3))
    ast = draw(fpy_funcdef(
        tuple(RealType() for _ in range(n)), return_type,
        grammar=ANF_PROFILE,
        max_depth=st.integers(3, 5),
        max_assigns=st.integers(1, 3),
        max_contexts=st.integers(0, 2),
        max_ifs=st.integers(0, 2),
        max_loops=st.just(0),
        max_whiles=st.integers(0, 1),
    ))
    args = [draw(value_for_type(RealType())) for _ in ast.args]
    return ast, args


def _run(ast, args):
    """``(result, None, None)`` or ``(None, exception name, message)`` -- FPy has
    undefined behavior, so a program that raises must raise the same way after
    the pass."""
    try:
        return repr(fp.Function(ast)(*args)), None, None
    except Exception as e:                       # noqa: BLE001
        return None, type(e).__name__, str(e)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(_program())
def test_the_passes_preserve_generated_programs(program) -> None:
    ast, args = program
    before = _run(ast, args)

    hoisted = Hoistable.apply(ast)
    left = [w for _e, w in Hoistable.refusals(hoisted)]
    assert not left, f'{left[0]}\n{hoisted.format()}'
    assert Hoistable.apply(hoisted).format() == hoisted.format(), (
        f'Hoistable is not idempotent\n{hoisted.format()}'
    )
    assert _run(hoisted, args) == before, (
        f'Hoistable changed semantics\n--- before ---\n{ast.format()}'
        f'\n--- after ---\n{hoisted.format()}'
    )

    out = ANF.apply(hoisted)
    dangerous = [w for _e, w in ANF.refusals(out) if w in _DANGEROUS]
    assert not dangerous, f'{dangerous[0]}\n{out.format()}'
    assert ANF.apply(out).format() == out.format(), f'ANF is not idempotent\n{out.format()}'
    assert _run(out, args)[:2] == before[:2], (
        f'ANF changed semantics\n--- before ---\n{ast.format()}'
        f'\n--- after ---\n{out.format()}'
    )
