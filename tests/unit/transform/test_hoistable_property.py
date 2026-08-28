"""Property tests for :class:`fpy2.transform.Hoistable` over generated programs.

``test_hoistable.py`` pins shapes on hand-written programs and
``test_hoistable_profile.py`` measures the corpus; neither goes deep.  Generation
reaches the nestings those two only sample.

Four properties, all on the same draw:

1. **It applies.**  ``Hoistable.apply`` runs the syntax checker itself, so this
   also asserts the output is a well-formed program.
2. **The invariant.**  ``refusals`` is empty: a temporary may be hoisted out of
   anywhere in the output, however deeply the lowerings nested.  Empty outright
   rather than empty of the dangerous reasons, since
   :data:`~tests.unit.generators.profiles.ANF_PROFILE` draws no comprehension.
3. **Idempotence.**  A second application changes nothing.
4. **Semantics.**  The interpreter agrees before and after, *including on which
   exception it raises* -- the property that catches an ordering regression,
   where a lowering hoisted above a left operand runs the operands out of turn.

``max_depth`` is 3--5, not lower: at depth 2 a ternary's arms are usually
leaves, so it is already in normal form and few draws lower one.

Generation cannot reach a lowering inside a rotated condition -- the generator's
loop template has a pure condition by construction -- so that composition is
covered by hand in ``test_hoistable.py::TestRotation``.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fpy2 as fp
from fpy2.transform import Hoistable
from fpy2.types import BoolType, RealType

from ..generators.fpy_program import fpy_funcdef, value_for_type
from ..generators.profiles import ANF_PROFILE


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
    """``(result, None)`` or ``(None, exception name)`` -- FPy has undefined
    behavior, so a program that raises must raise the same way after the pass."""
    try:
        return repr(fp.Function(ast)(*args)), None
    except Exception as e:                       # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(_program())
def test_hoistable_preserves_generated_programs(program) -> None:
    ast, args = program
    out = Hoistable.apply(ast)

    left = [w for _e, w in Hoistable.refusals(out)]
    assert not left, f'{left[0]}\n{out.format()}'

    assert Hoistable.apply(out).format() == out.format(), (
        f'not idempotent\n{out.format()}'
    )

    assert _run(ast, args) == _run(out, args), (
        f'semantics changed\n--- before ---\n{ast.format()}'
        f'\n--- after ---\n{out.format()}'
    )
