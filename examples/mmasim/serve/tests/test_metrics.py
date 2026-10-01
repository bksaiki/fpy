"""The local metrics against outputs whose errors are known exactly."""

import math

import pytest
import torch
from core import metrics, quant

M, N, K = 8, 17, 4
"""Ones `[M, K] @ [K, N]`: every exact product 4, every `|x|ᵀ|w|` 4."""


def _errors(sign: int) -> torch.Tensor:
    """Each output's error `[M, N]`: 16 exact, 8 of 4·2^-3 (a code moves under
    BF16 and FP8), the rest 4·2^-j for j in 10-16; signs alternate, or all
    *sign* for the large ones."""
    e = torch.zeros(M * N, dtype=torch.float64)
    js = [10 + i % 7 for i in range(M * N - 24)]
    e[16:16 + len(js)] = torch.tensor([(-1) ** i * 4 * 2.0 ** -j for i, j in enumerate(js)])
    e[-8:] = sign * 0.5
    return e.view(M, N)


def _measure(scheme: str, e: torch.Tensor) -> dict[str, float]:
    s = quant.SCHEMES[scheme]
    a, w = torch.ones(M, K), torch.ones(N, K)
    got = (4 + e).float()
    assert torch.equal(got.double(), 4 + e)    # every error is exact in FP32
    st = metrics.Stats()
    metrics.local([(st, got, None)], metrics.METRICS, quant.Quantized(s.x, a), quant.Quantized(s.w, w),
                  (a, w))
    return st.report(metrics.METRICS)


@pytest.mark.parametrize('scheme, sign', [('bf16', 1), ('bf16', -1), ('fp8-row', -1)])
def test_metrics_are_the_known_errors(scheme: str, sign: int) -> None:
    """Each metric equals its definition computed directly, and `flips`
    counts exactly the 8 large errors (downward under `fp8-row`, so that each
    row's maximum, and with it its scale, stays)."""
    e = _errors(sign)
    got = _measure(scheme, e)
    rel = (e / 4).flatten()
    nonzero = rel[rel != 0].abs()
    logs = torch.log2(nonzero).sort().values
    want = {
        'normwise': math.sqrt(float((e ** 2).sum()) / (16 * M * N)),
        'backward': float(logs.mean()),
        'backward_max': float(logs[-1]),
        'backward_exact': 1 - len(nonzero) / (M * N),
        'backward_normwise': 0.5 * math.log2(float((e ** 2).sum()) / (16 * M * N)),
        'ulp': float(torch.log2(1 + e.abs() / 2.0 ** -21).mean()),
        'rounded': float((e == 0).double().mean()),
        'bias': float(rel.mean()),
        'magnitude_bias': float(rel.mean()),
        'flips': 8 / (M * N),
    }
    for key, v in want.items():
        assert got[key] == pytest.approx(v, rel=1e-9, abs=1e-12), key
    for p in metrics.PERCENTILES:
        true = float(logs[math.ceil(len(logs) * p / 100) - 1])
        assert abs(got[f'backward_p{p}'] - true) <= 1 / 16, p    # a bin's middle


def test_a_scale_a_hair_apart_moves_no_code() -> None:
    """Under `fp8-row` an output nudged an ulp everywhere rescales its rows a
    hair, yet keeps every code: no flips."""
    got = _measure('fp8-row', torch.full((M, N), 4 * 2.0 ** -21, dtype=torch.float64))
    assert got['flips'] == 0 and got['rounded'] == 0
