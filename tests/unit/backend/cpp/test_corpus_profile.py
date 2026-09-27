"""What the corpus drives the emitter to do, pinned.

One sweep compiles every corpus function under FP64 and feeds three gates no
correctness test can see.

**Internal invariants.**  The emitter refuses for three unrelated reasons -- a
shape it does not implement, a program C++ cannot represent, and an invariant an
earlier phase was supposed to guarantee.  The third is a *backend bug*, and it
used to be spelled exactly like the other two, so an analysis producing
something structurally impossible read to the user as "your program is
unsupported."  :class:`CppInternalError` names that third kind, and the sweep
fails when any program reaches one, in either direction -- a real analysis bug,
or a refusal that was misclassified as internal and is actually reachable.  One
site is deliberately left as :class:`CppEmitError` -- *cannot dispatch X under
symbolic context*, where reachability could not be settled cheaply, and calling
a real refusal a compiler bug is the worse error.

**Where the emitter invents a name.**  ``_bind_operand`` mints a temporary where
the emitter reads an operand more than once and the operand is not already a
name.  The count is the thing that moves when a pass stops flattening the
program ahead of codegen.  Each site mints for a reason of its own: a *cast*
result (``_emit_ieee_min_max`` binds ``static_cast<double>(a)``, not an
identifier however atomic ``a`` is), a dimension read once per fixed-size layer
(``_emit_empty``), or an **aggregate** the site traverses twice (a list, a
slice, a tuple).

**How much keeps a handle.**  A program that boxes something it did not need to
still gives the right answer, just slower, so a precision regression is
invisible to the differential harness -- which is how a seeding bug that boxed
the inner level of every three-deep literal survived a full review.  This fails
when a change boxes something that used to be a value, and when a change unboxes
something new without anyone noticing: the second is not a bug, but it should be
a decision rather than a surprise.

Update a pin deliberately, with the reason in the commit message.
"""

import sys
from dataclasses import dataclass, field

import pytest

import fpy2 as fp
from fpy2.backend.cpp import emitter as _emitter
from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.emitter import CppEmitError, CppInternalError
from fpy2.backend.cpp.types import CppList, CppTuple
from fpy2.backend.cpp.unbox import UnboxMode
from tests.infra.backend.cpp import _inst_type, corpus

EXPECTED_COMPILED = 221
"""Corpus functions that compile.  A mint count only means something while this
holds -- fewer programs is fewer opportunities to mint."""

EXPECTED_MINTS = {
    '_convert_storage': 1,     # a tuple read field by field
    '_emit_empty': 28,         # a dimension, read once per fixed-size layer
    '_emit_ieee_min_max': 7,   # a cast result, not a nested operand
    '_emit_sum': 3,            # the list being folded
    '_integral_one_call': 5,   # the value made integral before the cast
    '_list_range': 4,          # the list being iterated
    '_visit_list_slice': 1,    # the list being sliced
}
"""Emitter sites that invent a name, and how often, over the corpus.

There is no `_emit_zip` entry because the `zip` / `enumerate` unfolds run
ahead of codegen: the lists a `zip` traversed twice are a comprehension's, so
the allocation is where the mint happens.

A dimension the *type* already carries is not bound, since the `std::array`
spells it and nothing reads the name; the corpus's only fixed-length
allocations are why `_emit_empty` is not higher, the rest being `std::vector`,
whose constructor does read the dimension.  A dimension `ConstFold` resolved
to a literal is not bound either -- that is one of ``test_list_comp5``'s.
"""

EXPECTED_LEVELS = 175
"""Every list level of every emitted signature in the corpus."""

EXPECTED_BOXED = 0
"""How many of those keep a handle."""


def _internal_cause(exc: BaseException) -> CppInternalError | None:
    """The :class:`CppInternalError` in *exc*'s cause chain, if any.

    ``CppCompiler`` wraps emitter errors with ``raise ... from e``, so the
    original type survives on ``__cause__`` even though the outer type does not
    distinguish them.
    """
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, CppInternalError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


def _retype(ty, fmt):
    """*ty* with every real leaf at *fmt*."""
    if isinstance(ty, fp.types.RealType):
        return fp.types.RealType(fmt)
    if isinstance(ty, fp.types.ListType):
        return fp.types.ListType(_retype(ty.elt, fmt))
    if isinstance(ty, fp.types.TupleType):
        return fp.types.TupleType(*(_retype(e, fmt) for e in ty.elts))
    return ty


def _levels(ty, path=''):
    """``(path, boxed)`` for each list level, following tuple fields."""
    if isinstance(ty, CppTuple):
        return [
            lv for i, e in enumerate(ty.elts) for lv in _levels(e, f'{path}.{i}')
        ]
    if isinstance(ty, CppList):
        return [(path or '0', ty.boxed)] + _levels(ty.elt, path + '>')
    return []


@dataclass
class _Sweep:
    compiled: int = 0
    mints: dict[str, int] = field(default_factory=dict)
    hits: list[str] = field(default_factory=list)
    """corpus programs that reached a backend invariant"""
    levels: int = 0
    boxed: list[str] = field(default_factory=list)
    """signature list levels that keep a handle"""


def _compile(out: _Sweep, f: fp.Function, arg_types) -> bool:
    """Whether *f* compiles under FP64, recording a backend invariant it hits."""
    try:
        CppCompiler().compile(f, ctx=fp.FP64, arg_types=arg_types)
        return True
    except Exception as e:
        internal = _internal_cause(e)
        if internal is not None:
            out.hits.append(f'{f.name}: {internal}')
        return False


def _sweep() -> _Sweep:
    out = _Sweep()
    counting = False
    original = _emitter.CppEmitter._bind_operand

    def bind_operand(self, expr):
        bound = original(self, expr)
        if counting and bound is not expr:
            caller = sys._getframe(1).f_code.co_name
            out.mints[caller] = out.mints.get(caller, 0) + 1
        return bound

    _emitter.CppEmitter._bind_operand = bind_operand
    try:
        for f in corpus():
            try:
                ty = fp.analysis.TypeInfer.check(f.ast)
                uniform = [_inst_type(t) for t in ty.arg_types]
            except Exception:
                continue       # already unsupported; not this test's business
            counting = True
            out.compiled += _compile(out, f, uniform)
            counting = False
            # every real/list-of-real argument at FP32 under the FP64 body, which
            # drives a narrower value into a wider place -- the
            # storage-reconciliation paths a uniform sweep never reaches
            _compile(out, f, [_retype(a, fp.FP32) for a in uniform])
            # under ALLOW: STRICT refuses a function that keeps a handle, which
            # would drop it here rather than count it
            try:
                params, ret = CppCompiler(unbox=UnboxMode.ALLOW).signature(
                    f, ctx=fp.FP64, arg_types=uniform,
                )
            except Exception:
                continue
            for kind, cty in [('arg', p) for p in params] + [('ret', ret)]:
                for path, is_boxed in _levels(cty):
                    out.levels += 1
                    if is_boxed:
                        out.boxed.append(f'{f.name}.{kind}[{path}]')
    finally:
        _emitter.CppEmitter._bind_operand = original
    return out


@pytest.fixture(scope='module')
def sweep() -> _Sweep:
    return _sweep()


# ----------------------------------------------------------------------
# Internal invariants


def test_no_corpus_program_trips_an_internal_invariant(sweep):
    """A hit is a bug in `format_infer` / `storage_infer` / `context_use`.

    Not in the program that exposed it -- these are conditions the emitter is
    entitled to assume, so the fix belongs upstream.  If a hit turns out to be
    genuinely reachable by a legal program, the site was misclassified: move it
    back to `CppEmitError` and record why in the audit doc.
    """
    assert not sweep.hits, (
        f'{len(sweep.hits)} corpus program(s) reached a backend invariant:\n  '
        + '\n  '.join(sweep.hits[:20])
    )


def test_the_internal_error_is_distinguishable():
    """The point of the split: the two kinds are told apart programmatically.

    Pinned because a subclass is easy to collapse back by accident -- catching
    `CppEmitError` still catches this, which is deliberate, so only the type
    itself carries the distinction.
    """
    err = CppInternalError('the storage ladder handed back a tuple')
    assert isinstance(err, CppEmitError)      # existing handlers still catch it
    assert 'internal error' in str(err)       # ...and a user can tell
    assert _internal_cause(err) is err

    plain = CppEmitError('unsupported for-loop target')
    assert _internal_cause(plain) is None

    # the wrapping preserves the type on the cause chain
    try:
        try:
            raise err
        except CppEmitError as e:
            raise RuntimeError('compilation failed') from e
    except RuntimeError as outer:
        assert _internal_cause(outer) is err


# ----------------------------------------------------------------------
# Where the emitter invents a name


def test_the_corpus_still_compiles(sweep):
    assert sweep.compiled == EXPECTED_COMPILED


def test_the_mint_counts_are_what_they_were(sweep):
    """Every site above mints for a reason of its own, so a *new* one is the
    emitter inventing a place it did not need to."""
    assert set(sweep.mints) <= set(EXPECTED_MINTS), (
        f'new site(s) minting a temporary: {set(sweep.mints) - set(EXPECTED_MINTS)}'
    )
    assert sweep.mints == EXPECTED_MINTS


# ----------------------------------------------------------------------
# How much keeps a handle


def test_no_signature_keeps_a_handle_unexpectedly(sweep):
    assert len(sweep.boxed) == EXPECTED_BOXED, (
        f'{len(sweep.boxed)} signature list levels keep a handle:\n  '
        + '\n  '.join(sweep.boxed)
        + '\n\nEach is either real sharing — in which case record it in '
          'docs/todos/backend-cpp.md and update EXPECTED_BOXED — or a '
          'precision regression.  Do not update the constant without deciding '
          'which.'
    )


def test_the_corpus_is_the_size_this_was_measured_against(sweep):
    """A guard on the guard: if the corpus shrinks, an empty result above stops
    meaning anything."""
    assert sweep.levels == EXPECTED_LEVELS, (
        f'the corpus has {sweep.levels} signature list levels, not '
        f'{EXPECTED_LEVELS}.  If that is intended, update the constant; if it '
        'dropped, the check above is weaker than it looks.'
    )
