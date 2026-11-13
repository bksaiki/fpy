import cases as cs
import pytest
import run
import torch


def test_summary_separates_lengths():
    rows = [{'design': 'example', 'implementation': 'fpy', 'batch_size': 2, 'dot_length': n,
             'threads': 1, 'seconds_per_call': 0.5} for n in (1024, 2048)]
    out = run.summarize(rows)
    assert [r['dot_length'] for r in out] == [1024, 2048]
    assert all(r['dots_per_second'] == 4 for r in out)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_validation_equates_signed_zeros_and_reports_first_mismatch(dtype):
    want = torch.tensor([1.0, -0.0, 2.0], dtype=dtype)
    v = cs.validate(cs.Prepared(lambda: torch.tensor([1.0, 0.0, 2.0], dtype=dtype), lambda: want, lambda: want))
    assert v['valid']
    got = torch.tensor([1.0, 0.0, 3.0], dtype=dtype)
    v = cs.validate(cs.Prepared(lambda: got, lambda: want, lambda: want))
    assert not v['valid']
    assert v['reason'] == 'output bit mismatch'
    assert v['mismatches'] == 1
    assert v['first_difference'] == 2
