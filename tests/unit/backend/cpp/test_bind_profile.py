"""Where the emitter still has to invent a name, pinned.

:class:`fpy2.transform.ANF` exists so the emitter never needs a place for an
operand: after it, every operand *it names* is already a name, and
``_bind_operand`` hands its argument straight back.  Where it still mints a
temporary, something reached the emitter un-flattened -- so this number is the
one that says how much of that job is left, and no correctness test can see it.

Every mint left is an **aggregate** operand (a list, a slice, a ``zip``) or a
*cast* result: ``_emit_ieee_min_max`` binds ``static_cast<double>(a)``, which is
not an identifier however atomic ``a`` is.  Aggregates are the follow-on --
naming one gives it a storage place, and a second place is what decides whether
a list keeps its shared handle.  When that lands, these counts drop, and the
drop should be a decision rather than a surprise.
"""

import importlib
import inspect

import pytest

import fpy2 as fp

from fpy2.backend.cpp import emitter as _emitter
from fpy2.backend.cpp.compiler import CppCompiler
from tests.infra.backend.cpp import _inst_type, _library_ignore
from tests.infra.examples import all_example_tests, all_unit_tests

EXPECTED_COMPILED = 201
"""Corpus functions that compile.  A mint count only means something while this
holds -- fewer programs is fewer opportunities to mint."""

EXPECTED_MINTS = {
    '_emit_ieee_min_max': 3,   # a cast result, not a nested operand
    '_emit_sum': 3,            # the list being folded
    '_emit_zip': 4,            # the lists being zipped
    '_list_range': 5,          # the list being iterated
    '_visit_list_slice': 1,    # the list being sliced
}
"""Emitter sites that still invent a name, and how often, over the corpus.

Absent from this table means never: ``_emit_empty`` and ``_emit_enumerate`` call
``_bind_operand`` but no corpus program gives either a compound operand.  That
is *unexercised*, not dead -- neither may be deleted on this evidence.
"""


def _corpus():
    yield from all_unit_tests()
    yield from all_example_tests()
    for name in ('core', 'eft', 'vector', 'matrix'):
        mod = importlib.import_module(f'fpy2.libraries.{name}')
        for f in mod.__dict__.values():
            if isinstance(f, fp.Function) and f.name not in _library_ignore:
                yield f


def _profile():
    """``(compiled, {site: mints})`` over the corpus."""
    mints: dict[str, int] = {}
    original = _emitter.CppEmitter._bind_operand

    def counting(self, expr):
        out = original(self, expr)
        if out is not expr:
            caller = inspect.stack()[1].function
            mints[caller] = mints.get(caller, 0) + 1
        return out

    _emitter.CppEmitter._bind_operand = counting
    compiled = 0
    try:
        for f in _corpus():
            try:
                ty = fp.analysis.TypeInfer.check(f.ast)
                args = [_inst_type(t) for t in ty.arg_types]
                CppCompiler().compile(f, ctx=fp.FP64, arg_types=args)
            except Exception:
                continue       # already unsupported; not this test's business
            compiled += 1
    finally:
        _emitter.CppEmitter._bind_operand = original
    return compiled, mints


@pytest.fixture(scope='module')
def profile():
    return _profile()


def test_the_corpus_still_compiles(profile):
    compiled, _ = profile
    assert compiled == EXPECTED_COMPILED


def test_no_scalar_operand_needs_a_name(profile):
    """The claim ANF is wired in for: an operand it names arrives as a name.

    Every site left mints for a reason this pass does not address -- an
    aggregate operand, or a cast -- so a *new* site here means something scalar
    reached the emitter nested.
    """
    _compiled, mints = profile
    assert set(mints) <= set(EXPECTED_MINTS), (
        f'new site(s) minting a temporary: {set(mints) - set(EXPECTED_MINTS)}'
    )


def test_the_mint_counts_are_what_they_were(profile):
    _compiled, mints = profile
    assert mints == EXPECTED_MINTS
