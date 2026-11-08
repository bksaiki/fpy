import cases as cs
import pytest
import run
import torch


def test_every_compiled_design_has_a_reference(manifest, cases):
    assert len(cases) == len(manifest['models'])
    assert len({c.name for c in cases}) == len({c.op for c in cases}) == len(cases)


@pytest.mark.parametrize('distribution', list(cs.DISTRIBUTIONS))
def test_bitwise_equivalent(cases, distribution):
    failures = {c.name: v for i, c in enumerate(cases)
                if not (v := cs.validate(cs.prepare(c, 4, 1000 + i, distribution=distribution)))['valid']}
    assert not failures


@pytest.mark.parametrize('length', [1024, 2048, 8192])
def test_lengths_and_repeat_calls(cases, length):
    for c in cases:
        p = cs.prepare(c, 2, 123, length)
        assert cs.validate(p)['valid'], c.name
        assert torch.equal(p.fpy(), p.fpy()), c.name
        assert torch.equal(p.mmasim(), p.mmasim()), c.name


def test_threads_agree(cases):
    # enough rows that `runners.h` splits them over every thread
    threads = torch.get_num_threads()
    for c in cases:
        p = cs.prepare(c, 1031, 7, 4 * c.k)
        many = p.fpy()
        torch.set_num_threads(1)
        try:
            one = p.fpy()
        finally:
            torch.set_num_threads(threads)
        assert torch.equal(many.view(cs.BITS[many.dtype]), one.view(cs.BITS[one.dtype])), c.name


def test_batched_calls_match_one_row_at_a_time(cases):
    for c in cases:
        # distinct rows, accumulators and scales catch row offsets, and rows
        # sharing an accumulator or scale
        a = torch.arange(3 * c.k, dtype=c.compute).reshape(3, c.k) / 128
        b = torch.full_like(a, 0.5)
        acc = torch.tensor([0.0, 1.0, -2.0], dtype=c.compute)
        scales = ()
        if c.scale_block:
            e = torch.arange(3 * (c.k // c.scale_block)).reshape(3, -1) % 5 - 2
            scales = tuple(s.to(c.compute) for s in (2.0 ** e, 2.0 ** e.flip(-1)))
        dpa = getattr(torch.ops.fpy2_models, c.op)
        batched = dpa(a, b, acc, *scales)
        for row in range(3):
            assert batched[row].item() == dpa(a[row], b[row], acc[row], *(s[row] for s in scales)).item()
        with pytest.raises(ValueError):
            cs.prepare(c, 1, 1, length=c.k + 1)


def test_summary_separates_lengths():
    rows = [{'design': 'example', 'implementation': 'fpy', 'batch_size': 2, 'dot_length': n,
             'threads': 1, 'seconds_per_call': 0.5} for n in (1024, 2048)]
    out = run.summarize(rows)
    assert [r['dot_length'] for r in out] == [1024, 2048]
    assert all(r['dots_per_second'] == 4 for r in out)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_validation_keeps_signed_zero_and_reports_first_mismatch(dtype):
    got = torch.tensor([1.0, 0.0], dtype=dtype)
    want = torch.tensor([1.0, -0.0], dtype=dtype)
    v = cs.validate(cs.Prepared(lambda: got, lambda: want))
    assert not v['valid']
    assert v['reason'] == 'output bit mismatch'
    assert v['mismatches'] == 1
    assert v['first_difference'] == 1
