"""
`scoring.evaluate` on a small random Qwen3.  Needs a GPU and
`transformers`; skipped without either.

    pytest serve/tests
"""

import math

import pytest
import torch

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

from core import metrics, quant, scoring, swap, workloads


def test_the_baseline_is_its_own_reference(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """R0 against itself is no distance at all, and its perplexity is the
    model's cross-entropy over the segments; another run is some distance."""
    run = swap.patch(model)
    segs = workloads.segments(torch.cat([tokens, tokens.flip(-1)], -1), context=16)
    totals = scoring.evaluate(model, run, segs, ['bf16-exact'])
    r0, r1 = totals['fp32'].report(), totals['bf16-exact'].report()
    assert (r0['kl'], r0['top1'], r0['rms_dp']) == (0.0, 1.0, 0.0)
    with torch.no_grad():
        nll = torch.cat([torch.nn.functional.cross_entropy(
            model(s).logits[0, :-1], s[0, 1:], reduction='none') for s in segs])
    assert math.isclose(r0['ppl'], math.exp(nll.mean().item()), rel_tol=1e-6)
    assert r0['tokens'] == r1['tokens'] == 2 * 15
    assert r1['kl'] > 0
    assert run.mode == 'fp32'


def test_a_quantizing_scheme_runs_end_to_end(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """Under `fp8-row` (`lm_head` left to R0, as `checkpoints.for_scheme`
    does), the exact run is farther from R0 than `bf16-exact`, and every
    design runs."""
    run = swap.patch(model)
    segs = workloads.segments(torch.cat([tokens, tokens.flip(-1)], -1), context=16)
    bf16 = scoring.evaluate(model, run, segs, ['bf16-exact'])['bf16-exact'].report()['kl']
    fp8 = quant.SCHEMES['fp8-row']
    swap.give(run, model, fp8, {}, ignore=['lm_head'])
    totals = scoring.evaluate(model, run, segs, swap.modes(fp8)[1:])
    assert totals['fp8-row-exact'].report()['kl'] > bf16
    assert all(0 < t.report()['kl'] < math.inf for m, t in totals.items() if m != 'fp32')
    swap.give(run, model, swap.BF16, {})


def test_runs_pair_against_r0_and_the_exact_run(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """Every run is paired against R0, a design against `bf16-exact` too; the
    paired standard error of Δ NLL is below the two runs' separate ones
    combined."""
    run = swap.patch(model)
    ids = torch.cat([tokens, tokens.flip(-1), tokens.roll(3, -1), tokens.roll(7, -1)], -1)
    totals = scoring.evaluate(model, run, workloads.segments(ids, context=16),
                                 ['amd.cdna2.bf16', 'bf16-exact'])
    assert list(totals) == ['fp32', 'bf16-exact', 'amd.cdna2.bf16']
    got = scoring.against(totals)
    assert set(got['fp32']) == {'bf16-exact', 'amd.cdna2.bf16'}
    assert set(got['bf16-exact']) == {'amd.cdna2.bf16'}
    r = got['fp32']['amd.cdna2.bf16']
    apart = math.hypot(metrics.paired(totals['amd.cdna2.bf16'].nll_seg)[1],
                       metrics.paired(totals['fp32'].nll_seg)[1])
    assert 0 < r['dnll_se'] < apart and r['kl'] > 0 and 0 <= r['p_holm'] <= 1

