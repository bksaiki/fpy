"""Where the emitter has to invent a name, pinned.

``_bind_operand`` mints a temporary where the emitter reads an operand more than
once and the operand is not already a name.  No correctness test can see the
count, and it is the thing that moves when a pass stops flattening the program
ahead of codegen -- so it is pinned, and a change to it should be a decision
rather than a surprise.

Each site mints for a reason of its own: a *cast* result
(``_emit_ieee_min_max`` binds ``static_cast<double>(a)``, not an identifier
however atomic ``a`` is), a dimension read once per fixed-size layer
(``_emit_empty``), or an **aggregate** the site traverses twice (a list, a
slice, a ``zip``).
"""

import importlib
import inspect

import pytest

import fpy2 as fp

from fpy2.backend.cpp import emitter as _emitter
from fpy2.backend.cpp.compiler import CppCompiler
from tests.infra.backend.cpp import _inst_type, _library_ignore
from tests.infra.examples import all_example_tests, all_unit_tests

EXPECTED_COMPILED = 207
"""Corpus functions that compile.  A mint count only means something while this
holds -- fewer programs is fewer opportunities to mint."""

EXPECTED_MINTS = {
    '_emit_empty': 28,         # a dimension, read once per fixed-size layer
    '_emit_ieee_min_max': 6,   # a cast result, not a nested operand
    '_emit_sum': 3,            # the list being folded
    '_emit_zip': 4,            # the lists being zipped
    '_list_range': 4,          # the list being iterated
    '_visit_list_slice': 1,    # the list being sliced
}
"""Emitter sites that invent a name, and how often, over the corpus.

Absent from this table means never: ``_emit_enumerate`` calls ``_bind_operand``
but no corpus program gives it a compound operand.  That is *unexercised*, not
dead -- it may not be deleted on this evidence.
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


def test_no_new_site_mints(profile):
    """Every site above mints for a reason of its own, so a *new* one is the
    emitter inventing a place it did not need to."""
    _compiled, mints = profile
    assert set(mints) <= set(EXPECTED_MINTS), (
        f'new site(s) minting a temporary: {set(mints) - set(EXPECTED_MINTS)}'
    )


def test_the_mint_counts_are_what_they_were(profile):
    _compiled, mints = profile
    assert mints == EXPECTED_MINTS
