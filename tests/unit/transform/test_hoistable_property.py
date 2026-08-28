"""Property tests for :class:`fpy2.transform.Hoistable` over generated programs.

``test_hoistable.py`` pins shapes on hand-written programs and
``test_hoistable_profile.py`` measures the corpus; neither goes deep.  The cases
where one lowering meets another -- a ternary inside a rotated condition, a chain
inside a ternary arm -- are only reachable by generating them.

Four properties, all on the same draw:

1. **It applies.**  ``Hoistable.apply`` runs the syntax checker itself, so this
   also asserts the output is a well-formed program.
2. **The invariant.**  ``refusals`` is empty: a temporary may be hoisted out of
   anywhere in the output, however deeply the lowerings nested.  Empty outright
   rather than empty of the dangerous reasons, since
   :data:`~tests.unit.generators.profiles.ANF_PROFILE` draws no comprehension.
3. **Idempotence.**  A second application changes nothing.
4. **Semantics.**  The interpreter agrees before and after, *including on which
   exception it raises*.  This is the property that catches an ordering
   regression: a lowering hoisted above a left operand runs the operands out of
   turn, and the two raise from different places.

**What the draws reach.**  ``max_depth`` is 3--5 rather than
``test_anf_property.py``'s 2--4: at depth 2 a ternary's arms are almost always
leaves, so it is already in normal form and only one draw in 150 lowers one.  At
3--5 a 150-example run reaches ~120 lowerings, a third of them ternaries, with
~16 nesting one lowering inside another's arm.

Rotation is reached too, which ``test_anf_property.py`` says it does not: the
generator's loop template ``c = 0; while c < N`` has a *pure* condition, which
ANF declines to rotate, but ``c < N`` is not an atom and that is this pass's
gate.  What generation *cannot* reach is a lowering inside a rotated condition,
since that template's condition is pure by construction; the composition is
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
        return None, type(e).__name__


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
