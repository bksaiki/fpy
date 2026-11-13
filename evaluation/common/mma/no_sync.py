"""
MMA-Sim's arithmetic (`mmasim.arithmetic.fdpa`) without host
synchronizations, for the tuned MMA-Sim baselines (`gsm8k/mmasim_matmul.tune`,
`cpu_performance`).  Each function is MMA-Sim's own, line for line, with its
data-dependent Python branch (`pow2`) and its boolean-mask updates
(`x[mask] = ...`, which count the mask on the host) replaced by `torch.where`;
the results are bit for bit MMA-Sim's.  MMA-Sim's FMA chain (`MMA_FMA.dpa`, an
`fma` per product) runs as one kernel per call, the same `fma`s in the same
order (:func:`fma_dpa`).  :func:`install` patches them into the module, where
MMA-Sim's operations look them up.

Adapted from MMA-Sim (https://github.com/microsoft/MMA-Sim),
Copyright (c) Microsoft Corporation, under the MIT License.
"""

import functools

import torch
from mmasim.arithmetic import fdpa, fma, ftz_mul_add
from mmasim.arithmetic.helper import dtype_subnormal_exponent
from torch._inductor import lowering
from torch.utils.cpp_extension import load_inline

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


def ftz_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # MMA-Sim's, without its FP16/BF16-only assert: BF16 values held in FP32
    # flush below the same 2^-126, so give the same results
    assert x.dtype == y.dtype
    x = ftz_mul_add.flush_subnormal(x, keep_sign=False)
    y = ftz_mul_add.flush_subnormal(y, keep_sign=False)
    return ftz_mul_add.flush_subnormal(x.float() * y.float(), keep_sign=True)


_CHAIN_CPU = r"""
#include <torch/extension.h>
#include <cmath>

template <typename T>
static void chain(const T *a, const T *b, T *o, int64_t M, int64_t N, int64_t K,
                  const int64_t *sa, const int64_t *sb) {
#pragma omp parallel for collapse(2) if (M * N > 1024)
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            T acc = o[m * N + n];
            for (int64_t k = 0; k < K; ++k)
                acc = std::fma(a[m * sa[0] + n * sa[1] + k * sa[2]],
                               b[m * sb[0] + n * sb[1] + k * sb[2]], acc);
            o[m * N + n] = acc;
        }
}

torch::Tensor fma_chain(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    auto o = c.clone();
    auto sa = a.strides(), sb = b.strides();
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "fma_chain", [&] {
        chain<scalar_t>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), o.data_ptr<scalar_t>(),
                        a.size(0), a.size(1), a.size(2), sa.data(), sb.data());
    });
    return o;
}
"""


@functools.cache
def _chain_cpu():
    return load_inline('fpy_eval_fma_chain', _CHAIN_CPU, functions=['fma_chain'],
                       extra_cflags=['-O3', '-fopenmp'], extra_ldflags=['-fopenmp'])


if torch.cuda.is_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _chain_gpu(a, b, c, o, M, N, K, sa0, sa1, sa2, sb0, sb1, sb2, BLOCK: tl.constexpr):
        p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = p < M * N
        m, n = p // N, p % N
        acc = tl.load(c + p, mask=mask)
        for k in range(K):
            x = tl.load(a + m * sa0 + n * sa1 + k * sa2, mask=mask)
            y = tl.load(b + m * sb0 + n * sb1 + k * sb2, mask=mask)
            # `tl.fma` (the hardware FMA, IEEE subnormals), not libdevice's,
            # which Triton flushes by default
            acc = tl.fma(x, y, acc)
        tl.store(o + p, acc, mask=mask)


def _broadcast(a: torch.Tensor, b: torch.Tensor,
               c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """*a* and *b* `[..., K]` broadcast to *c*'s shape, all three with one
    leading dimension if *c* is `[batch]`; *c* contiguous."""
    a, b = (x.expand(*c.shape, x.shape[-1]) for x in (a, b))
    if c.dim() == 1:
        a, b, c = a[None], b[None], c[None]
    return a, b, c.contiguous()


@torch.compiler.disable
def _fma_dpa_cpu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return _chain_cpu().fma_chain(*_broadcast(a, b, c)).view(c.shape)


def fma_dpa(self: fma.MMA_FMA, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """`MMA_FMA.dpa` as one kernel: `c = fma(a[..., i], b[..., i], c)` for each
    `i` in order, for *a* and *b* `[..., K]` (broadcast to *c*'s shape) and
    *c* `[m, n]` or `[batch]`.  On the GPU, a Triton kernel `torch.compile`
    compiles into its graph; on the CPU, C++ that Dynamo does not trace."""
    if not c.is_cuda:
        return _fma_dpa_cpu(a, b, c)
    shape = c.shape
    a, b, c = _broadcast(a, b, c)
    o = torch.empty_like(c)
    m, n, k = a.shape
    torch.library.wrap_triton(_chain_gpu)[(triton.cdiv(m * n, 256),)](
        a, b, c, o, m, n, k, *a.stride(), *b.stride(), BLOCK=256)
    return o.view(shape)


_FMA_DPA = fma.MMA_FMA.dpa

_PATCHED = {n: globals()[n] for n in _ORIGINAL}
_FTZ_MUL = ftz_mul_add.ftz_mul
_FREXP_LOWERINGS = {o: lowering.lowerings[o] for o in torch.ops.aten.frexp.op_overloads()
                    if o in lowering.lowerings}


def install(on: bool = True) -> None:
    """Patch MMA-Sim's arithmetic with these, and make it compile under
    `torch.compile` (or restore its own).  Inductor's vectorized CPU `frexp`
    of FP64 does not compile, so `aten.frexp` falls back to ATen."""
    for n, f in (_PATCHED if on else _ORIGINAL).items():
        setattr(fdpa, n, f)
    ftz_mul_add.ftz_mul = ftz_mul if on else _FTZ_MUL
    fma.MMA_FMA.dpa = fma_dpa if on else _FMA_DPA
    if on:
        lowering.make_fallback(torch.ops.aten.frexp)
    else:
        lowering.lowerings.update(_FREXP_LOWERINGS)
