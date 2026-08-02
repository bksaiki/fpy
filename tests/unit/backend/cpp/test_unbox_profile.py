"""How much of the corpus keeps a handle, pinned.

Every other gate on unboxing checks *correctness*: the differential harness
bit-compares against the interpreter, and a program that boxes something it did
not need to still gives the right answer, just slower.  So a precision
regression is invisible — which is how a seeding bug that boxed the inner level
of every three-deep literal survived a full review and got written off as
"genuine sharing".

This is the missing direction.  It fails when a change boxes something that used
to be a value, and it fails when a change unboxes something new without anyone
noticing — the second is not a bug, but it should be a decision rather than a
surprise, and the number here is what the soundness argument is measured against.
"""

import importlib

import pytest

import fpy2 as fp

from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.types import CppList, CppTuple
from tests.infra.backend.cpp import _inst_type, _library_ignore
from tests.infra.examples import all_example_tests, all_unit_tests

# Every list level of every emitted signature in the corpus, and how many keep
# a handle.  Update deliberately, with the reason in the commit message.
EXPECTED_LEVELS = 166
EXPECTED_BOXED = 0


def _corpus():
    yield from all_unit_tests()
    yield from all_example_tests()
    for name in ('core', 'eft', 'vector', 'matrix'):
        mod = importlib.import_module(f'fpy2.libraries.{name}')
        for f in mod.__dict__.values():
            if isinstance(f, fp.Function) and f.name not in _library_ignore:
                yield f


def _levels(ty, path=''):
    """``(path, boxed)`` for each list level, following tuple fields."""
    if isinstance(ty, CppTuple):
        return [
            lv for i, e in enumerate(ty.elts) for lv in _levels(e, f'{path}.{i}')
        ]
    if isinstance(ty, CppList):
        return [(path or '0', ty.boxed)] + _levels(ty.elt, path + '>')
    return []


def _profile():
    """``(total levels, [names that keep a handle])`` over the corpus."""
    total, boxed = 0, []
    for f in _corpus():
        try:
            ty = fp.analysis.TypeInfer.check(f.ast)
            args = [_inst_type(t) for t in ty.arg_types]
            params, ret = CppCompiler().signature(
                f, ctx=fp.FP64, arg_types=args,
            )
        except Exception:
            continue          # already unsupported; not this test's business
        for kind, cty in [('arg', p) for p in params] + [('ret', ret)]:
            for path, is_boxed in _levels(cty):
                total += 1
                if is_boxed:
                    boxed.append(f'{f.name}.{kind}[{path}]')
    return total, boxed


@pytest.fixture(scope='module')
def profile():
    return _profile()


def test_no_signature_keeps_a_handle_unexpectedly(profile):
    _total, boxed = profile
    assert boxed == [], (
        f'{len(boxed)} signature list levels keep a handle:\n  '
        + '\n  '.join(boxed)
        + '\n\nEach is either real sharing — in which case record it in '
          'docs/todos/unboxing-gaps.md and update EXPECTED_BOXED — or a '
          'precision regression.  Do not update the constant without deciding '
          'which.'
    )


def test_the_corpus_is_the_size_this_was_measured_against(profile):
    """A guard on the guard: if the corpus shrinks, an empty result above stops
    meaning anything."""
    total, _boxed = profile
    assert total == EXPECTED_LEVELS, (
        f'the corpus has {total} signature list levels, not {EXPECTED_LEVELS}. '
        'If that is intended, update the constant; if it dropped, the check '
        'above is weaker than it looks.'
    )
