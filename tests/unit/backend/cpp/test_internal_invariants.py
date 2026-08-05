"""No corpus program may trip a backend invariant.

The emitter refuses for three unrelated reasons — a shape it does not implement,
a program C++ cannot represent, and an invariant an earlier phase was supposed to
guarantee.  The third is a *backend bug*, and it used to be spelled exactly like
the other two, so an analysis producing something structurally impossible read to
the user as "your program is unsupported."

:class:`CppInternalError` names that third kind.  This is the gate that keeps the
distinction honest: it fails when any program in the corpus reaches one, in
either direction — a real analysis bug, or a refusal that was misclassified as
internal and is actually reachable.

One site is deliberately left as :class:`CppEmitError` -- *cannot dispatch X
under symbolic context*, where reachability could not be settled cheaply, and
calling a real refusal a compiler bug is the worse error.
"""

import importlib

import pytest

import fpy2 as fp

from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.emitter import CppInternalError
from tests.infra.backend.cpp import _inst_type, _library_ignore
from tests.infra.examples import all_example_tests, all_unit_tests

# The generated matrix's mixed-format instantiations, which are what reach the
# storage-reconciliation paths at all -- a uniform-format sweep never does.
_FORMATS = (fp.FP32, fp.FP64)


def _corpus():
    yield from all_unit_tests()
    yield from all_example_tests()
    for name in ('core', 'eft', 'vector', 'matrix'):
        mod = importlib.import_module(f'fpy2.libraries.{name}')
        for f in mod.__dict__.values():
            if isinstance(f, fp.Function) and f.name not in _library_ignore:
                yield f


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


def _attempt(func, arg_types) -> str | None:
    """Compile *func*, returning a failure description iff it hit an invariant."""
    try:
        CppCompiler().compile(func, arg_types=arg_types)
    except Exception as e:
        internal = _internal_cause(e)
        if internal is not None:
            return f'{func.name}: {internal}'
    return None


def _sweep():
    """Every corpus function, at uniform and at mixed argument formats."""
    hits: list[str] = []
    for f in _corpus():
        try:
            ty = fp.analysis.TypeInfer.check(f.ast)
        except Exception:
            continue
        # uniform: whatever the corpus harness would use
        try:
            uniform = [_inst_type(t) for t in ty.arg_types]
        except Exception:
            continue
        hit = _attempt(f, uniform)
        if hit is not None:
            hits.append(hit)
        # mixed: re-instantiate every real/list-of-real argument at FP32, which
        # is what drives a narrower value into a wider place
        for fmt in _FORMATS:
            try:
                mixed = [_inst_type(t) for t in ty.arg_types]
                mixed = [_retype(a, fmt) for a in mixed]
            except Exception:
                continue
            hit = _attempt(f, mixed)
            if hit is not None:
                hits.append(hit)
    return hits


def _retype(ty, fmt):
    """*ty* with every real leaf at *fmt*."""
    if isinstance(ty, fp.types.RealType):
        return fp.types.RealType(fmt)
    if isinstance(ty, fp.types.ListType):
        return fp.types.ListType(_retype(ty.elt, fmt))
    if isinstance(ty, fp.types.TupleType):
        return fp.types.TupleType(*(_retype(e, fmt) for e in ty.elts))
    return ty


@pytest.fixture(scope='module')
def hits():
    return _sweep()


def test_no_corpus_program_trips_an_internal_invariant(hits):
    """A hit is a bug in `format_infer` / `storage_infer` / `context_use`.

    Not in the program that exposed it -- these are conditions the emitter is
    entitled to assume, so the fix belongs upstream.  If a hit turns out to be
    genuinely reachable by a legal program, the site was misclassified: move it
    back to `CppEmitError` and record why in the audit doc.
    """
    assert not hits, (
        f'{len(hits)} corpus program(s) reached a backend invariant:\n  '
        + '\n  '.join(hits[:20])
    )


def test_the_internal_error_is_distinguishable():
    """The point of the split: the two kinds are told apart programmatically.

    Pinned because a subclass is easy to collapse back by accident -- catching
    `CppEmitError` still catches this, which is deliberate, so only the type
    itself carries the distinction.
    """
    from fpy2.backend.cpp.emitter import CppEmitError

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
