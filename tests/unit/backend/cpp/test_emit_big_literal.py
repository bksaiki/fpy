"""An integral literal too large for a C++ integer literal.

``_emit_numeric_literal`` printed any value with ``denominator == 1`` as decimal
digits, with no range check.  A decimal literal takes the first of ``int`` /
``long`` / ``long long`` that fits it and is ill-formed when none does -- gcc
accepts it with only *"integer constant is too large for its type"* and the value
becomes ``0``.

Storage selection refuses such a value for a *scalar*, so this was only reachable
where nothing asks: a list element or a slot.  ``zs = [1e300, y]`` compiled to a
301-digit token and returned ``0`` where the interpreter returns ``1e300``.

The fix routes both spellings -- ``_emit_numeric_literal`` and the
``_visit_integer`` shortcut, which printed ``str(e.val)`` directly -- through
``_value_cpp_type``, which is also what decides whether a literal argument needs
a cast (see ``test_emit_literal_types.py``).  "What type is this token" and "can
this token be written at all" are the same question.
"""

from fractions import Fraction

import fpy2 as fp
import pytest

from fpy2.ast.fpyast import Integer
from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.emitter import CppEmitError, CppEmitter, _value_cpp_type
from fpy2.types import RealType

_R64 = RealType(fp.FP64)


class _Probe(CppEmitter):
    """``_emit_numeric_literal`` in isolation -- it reads no emitter state."""

    def __init__(self):
        pass


def _emit(v) -> str:
    return _Probe()._emit_numeric_literal(Fraction(v), at=Integer(0, None))


class TestDigitsOnlyWhileALiteralHoldsIt:
    def test_ordinary_integers_still_print_as_digits(self):
        """The counterweight: the common case is untouched."""
        assert _emit(0) == '0'
        assert _emit(1) == '1'
        assert _emit(-1) == '-1'
        assert _emit(2 ** 53) == str(2 ** 53)

    def test_the_largest_spellable_integer_still_prints_as_digits(self):
        assert _emit(2 ** 63 - 1) == str(2 ** 63 - 1)
        assert _emit(-(2 ** 63 - 1)) == str(-(2 ** 63 - 1))

    def test_past_that_it_prints_as_a_float(self):
        """``2**63`` is a power of two, so binary64 holds it exactly."""
        out = _emit(2 ** 63)
        assert not out.lstrip('-').isdigit(), out
        assert float(out) == float(2 ** 63)

    def test_a_huge_non_double_is_refused_not_mangled(self):
        """No spelling exists, so refuse rather than emit something wrong.

        ``2**63 + 1`` is odd and above ``2**53``, so no ``double`` holds it and
        no integer literal is wide enough.  Refusing is the acceptable answer;
        emitting digits that gcc folds to ``0`` is not.
        """
        with pytest.raises(CppEmitError, match='not representable in'):
            _emit(2 ** 63 + 1)

    def test_a_non_dyadic_fraction_is_refused_the_same_way(self):
        """One message covers both: no C++ literal holds the value exactly, and
        the way out is the same -- round it to a format that does."""
        with pytest.raises(CppEmitError, match='not representable in'):
            _emit(Fraction(1, 3))


class TestReachableThroughAListSlot:
    """The path that made this a wrong answer rather than a refusal."""

    def test_a_big_literal_in_a_list_element(self):
        @fp.fpy(ctx=fp.FP64)
        def f(y: fp.Real) -> fp.Real:
            zs = [1e300, y]
            return zs[0]

        out = CppCompiler().compile(f, arg_types=[_R64])
        assert '1e+300' in out, out
        # the 301-digit token is gone
        assert '0000000000000000000000' not in out, out

    def test_a_big_literal_stored_into_a_slot(self):
        @fp.fpy(ctx=fp.FP64)
        def f(y: fp.Real) -> fp.Real:
            zs = [y, y]
            zs[0] = 1e300
            return zs[0]

        out = CppCompiler().compile(f, arg_types=[_R64])
        assert '1e+300' in out, out
        assert '0000000000000000000000' not in out, out

    def test_the_integer_shortcut_goes_through_the_same_path(self):
        """``_visit_integer`` used to print ``str(e.val)``, bypassing the check.

        Pinned because it is a second spelling of the same bug, and a plausible
        place to reintroduce it as an "obvious" fast path.
        """
        assert _value_cpp_type(Fraction(2 ** 100)) is None
        src = CppEmitter._visit_integer.__doc__ or ''
        assert 'str(e.val)' not in src.replace('`str(e.val)`', '')
