"""
MMA-Sim's arithmetic (`mmasim.arithmetic.fdpa`) without host
synchronizations, for the tuned MMA-Sim baseline (`mmasim_matmul.tune`).
Each function is MMA-Sim's own, line for line, with its data-dependent
Python branch (`pow2`) and its boolean-mask updates (`x[mask] = ...`, which
count the mask on the host) replaced by `torch.where`; the results are bit
for bit MMA-Sim's.  :func:`install` patches them into the module, where
MMA-Sim's operations look them up.

Adapted from MMA-Sim (https://github.com/microsoft/MMA-Sim),
Copyright (c) Microsoft Corporation, under the MIT License.
"""

import torch
from mmasim.arithmetic import fdpa
from mmasim.arithmetic.helper import dtype_subnormal_exponent

_ORIGINAL = {n: getattr(fdpa, n) for n in ('pow2', 'frexp_and_normalize', 'ldexp_and_normalize', 't_fdpa',
                                           'st_fdpa', 'gst_fdpa', 'tr_fdpa', 'gtr_fdpa')}


def pow2(e: torch.Tensor) -> torch.Tensor:
    # MMA-Sim widens to FP64 only when FP32 cannot hold 2^e; powers of two are
    # exact in both, so always FP64 gives the same values without a branch
    return torch.exp2(e.double())


def frexp_and_normalize(x: torch.Tensor,
                        e_subnormal: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dtype in dtype_subnormal_exponent, f"Unsupported dtype: {x.dtype}"
    if x.dtype == torch.float64:
        s, e = torch.frexp(x)
    else:
        s, e = torch.frexp(x.float())
    s, e = s * 2, e - 1
    if e_subnormal is None:
        e_subnormal = dtype_subnormal_exponent[x.dtype]
    subnormals = e < e_subnormal
    s = torch.where(subnormals, s * pow2(e - e_subnormal), s).to(s.dtype)
    e = torch.where(subnormals, e_subnormal, e)
    e = torch.where(s == 0.0, e_subnormal, e)
    return s, e


def ldexp_and_normalize(s: torch.Tensor, e: torch.Tensor, rho: str) -> torch.Tensor:
    if rho == "RNE-FP16":
        s, e = frexp_and_normalize(s * pow2(e), e_subnormal=-14)
        s = torch.round(s * 2.0**10) * 2.0**-10
        res = (s * pow2(e)).to(torch.float16)
    elif rho == "RNE-FP32":
        res = (s * pow2(e)).to(torch.float32)
    else:
        s, e = frexp_and_normalize(s * pow2(e), e_subnormal=-126)
        if rho == "RZ-E8M13":
            s = torch.trunc(s * 2.0**13) * 2.0**-13
        else:
            s = torch.trunc(s * 2.0**23) * 2.0**-23
        res = (s * pow2(e)).to(torch.float32)
    nans = res.isnan()
    if res.dtype == torch.float16:
        res = torch.where(nans, 0x7FFF, res.view(torch.int16)).to(torch.int16).view(torch.float16)
    else:
        res = torch.where(nans, 0x7FFFFFFF, res.view(torch.int32)).to(torch.int32).view(torch.float32)
    return res


def t_fdpa(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, F: int, rho: str,
           e_zero: int, e_subnormal_a: int | None = None,
           e_subnormal_b: int | None = None) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a, e_subnormal_a)
    s_b, e_b = frexp_and_normalize(b, e_subnormal_b)
    s_c, e_c = frexp_and_normalize(c)
    s = torch.cat([s_a * s_b, s_c.unsqueeze(-1)], dim=-1)
    e = torch.cat([e_a + e_b, e_c.unsqueeze(-1)], dim=-1)
    e = torch.where(s == 0.0, e_zero, e)
    sum, e_max = fdpa.truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho) + 0.0  # -0.0 to +0.0


def st_fdpa(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            F: int, rho: str, e_zero: int, e_subnormal_a: int | None = None,
            e_subnormal_b: int | None = None) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a, e_subnormal_a)
    s_b, e_b = frexp_and_normalize(b, e_subnormal_b)
    s_c, e_c = frexp_and_normalize(c)
    s_alpha, e_alpha = frexp_and_normalize(alpha)
    s_beta, e_beta = frexp_and_normalize(beta)
    s = torch.cat([s_a * s_b * s_alpha * s_beta, s_c.unsqueeze(-1)], dim=-1)
    e = torch.cat([e_a + e_b + e_alpha + e_beta, e_c.unsqueeze(-1)], dim=-1)
    e = torch.where(s == 0.0, e_zero, e)
    sum, e_max = fdpa.truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho) + 0.0  # -0.0 to +0.0


def gst_fdpa(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
             G: int, F: int, rho: str, e_zero: int) -> torch.Tensor:
    p = a.float() * b.float()
    p = p.view(*p.shape[:-1], -1, G)
    apply_e_zero = (p == 0.0).all(dim=-1)
    p = p.sum(dim=-1)
    K_block = a.shape[-1] // alpha.shape[-1]
    alpha = torch.repeat_interleave(alpha, K_block // G, dim=-1)
    beta = torch.repeat_interleave(beta, K_block // G, dim=-1)
    s_alpha, e_alpha = frexp_and_normalize(alpha)
    s_beta, e_beta = frexp_and_normalize(beta)
    s_scale, e_scale = s_alpha * s_beta, e_alpha + e_beta
    apply_e_zero |= s_scale == 0.0
    e_scale = torch.where(apply_e_zero, e_zero, e_scale)
    s_c, e_c = frexp_and_normalize(c)
    e_c = torch.where(s_c == 0.0, e_zero, e_c)
    s = torch.cat([p * s_scale, s_c.unsqueeze(-1)], dim=-1)
    e = torch.cat([e_scale, e_c.unsqueeze(-1)], dim=-1)
    sum, e_max = fdpa.truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho) + 0.0  # -0.0 to +0.0


def tr_fdpa(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, F: int, F2: int,
            rho: str) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s, e = s_a * s_b, e_a + e_b
    overflow = ((s.abs() >= 2.0) + e) >= 128
    s = torch.where(overflow, s * float("inf"), s)
    e = torch.where(s == 0.0, -999, e)
    s_dot, e_dot = fdpa.truncated_fused_sum(s, e, F)

    s_c, e_c = frexp_and_normalize(c)
    e_max = torch.max(e_dot, e_c)
    s_dot = torch.floor(s_dot * pow2(e_dot - e_max + (F2 + 1))) * 2.0 ** -(F2 + 1)
    s_c = torch.floor(s_c * pow2(e_c - e_max + F)) * 2.0**-F
    sum, e_max = frexp_and_normalize((s_dot + s_c) * pow2(e_max), e_subnormal=-126)
    sum = torch.floor(sum * 2.0**F2) * 2.0**-F2
    return ldexp_and_normalize(sum, e_max, rho)


def gtr_fdpa(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, F: int, F2: int,
             rho: str) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s = s_a * s_b
    e = e_a + e_b
    e = torch.where(s == 0.0, -999, e)
    s_even, e_even = fdpa.truncated_fused_sum(s[..., ::2], e[..., ::2], F)
    s_odd, e_odd = fdpa.truncated_fused_sum(s[..., 1::2], e[..., 1::2], F)

    e_dot = torch.max(e_even, e_odd)
    s_dot = torch.floor(s_even * pow2(e_even - e_dot + F)) * 2.0**-F
    s_dot += torch.floor(s_odd * pow2(e_odd - e_dot + F)) * 2.0**-F

    s_c, e_c = frexp_and_normalize(c)
    e_max = torch.max(e_dot, e_c)
    s_dot = torch.floor(s_dot * pow2(e_dot - e_max + (F2 + 1))) * 2.0 ** -(F2 + 1)
    s_c = torch.floor(s_c * pow2(e_c - e_max + F)) * 2.0**-F
    s_c = torch.where(e_c < e_max - F - 1, 0.0, s_c)
    s_dot, e_max = frexp_and_normalize((s_dot + s_c) * pow2(e_max), e_subnormal=-126)
    s_dot = torch.floor(s_dot * 2.0**F2) * 2.0**-F2
    return ldexp_and_normalize(s_dot, e_max, rho)


_PATCHED = {n: globals()[n] for n in _ORIGINAL}


def install(on: bool = True) -> None:
    """Patch MMA-Sim's arithmetic with these (or restore its own)."""
    for n, f in (_PATCHED if on else _ORIGINAL).items():
        setattr(fdpa, n, f)
