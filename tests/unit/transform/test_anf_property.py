"""Property tests for :class:`fpy2.transform.ANF` over generated programs.

``test_anf.py`` pins shapes on hand-written programs and ``test_anf_profile.py``
measures the corpus; neither goes deep.  The corpus is shallow -- the residue
measurement found nothing nested more than two levels -- so the cases where a
lowering meets another lowering are only reachable by generating them.

Four properties, all on the same draw:

1. **It applies.**  ``ANF.apply`` runs the syntax checker itself, so this also
   asserts the output is a well-formed program.
2. **No dangerous refusal.**  A ternary arm, a short-circuited operand and a
   ``while`` condition are the three positions the cpp emitter hoists out of
   (three miscompiles in ``docs/todos/backend-cpp.md``).  ``ANF`` no longer
   empties them -- :class:`~fpy2.transform.Hoistable` does, and ``ANF`` raises
   without it -- so the draw goes through that pass first and this asserts the
   pairing holds however deeply the lowerings nested.
3. **Idempotence.**  A second application changes nothing.
4. **Semantics.**  The interpreter agrees before and after, including on which
   exception it raises -- an eagerly evaluated ternary arm or short-circuited
   operand would raise where FPy returns, which is exactly witness 2 and 3.

Uses :data:`~tests.unit.generators.profiles.ANF_PROFILE`, which is skewed toward
``IF_EXPR`` and ``AND``/``OR``.  It does not reach ``while``-condition rotation:
the generator's loop template has a pure condition by construction, so no draw
can produce one needing a place.
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
        max_depth=st.integers(2, 4),
        max_assigns=st.integers(1, 3),
        max_contexts=st.integers(0, 2),
        max_ifs=st.integers(0, 2),
        max_loops=st.just(0),
        max_whiles=st.integers(0, 1),
    ))
    args = [draw(value_for_type(RealType())) for _ in ast.args]
    return ast, args


def _run(ast, args):
    """``(result, None)`` or ``(None, exception name)`` -- FPy has undefined
    behavior, so a program that raises must raise the same way after the pass."""
    try:
        return repr(fp.Function(ast)(*args)), None
    except Exception as e:                       # noqa: BLE001
        return None, type(e).__name__


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(_program())
def test_anf_preserves_generated_programs(program) -> None:
    ast, args = program
    out = ANF.apply(Hoistable.apply(ast))

    dangerous = [w for _e, w in ANF.refusals(out) if w in _DANGEROUS]
    assert not dangerous, f'{dangerous[0]}\n{out.format()}'

    assert ANF.apply(out).format() == out.format(), f'not idempotent\n{out.format()}'

    assert _run(ast, args) == _run(out, args), (
        f'semantics changed\n--- before ---\n{ast.format()}'
        f'\n--- after ---\n{out.format()}'
    )
