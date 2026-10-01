"""
Every `compile.DESIGNS` entry as a :class:`generator.Design`, by name.
"""

from dataclasses import replace

from compile import _CDNA3_FP8_PAIRS, _F8F6F4_PAIRS, _FP8_PAIRS
from models.nv import RNE_FP16, RZ_E8M13, RZ_FP32

import fpy2 as fp

from . import Design


def _t(L: int, a: fp.EFloatContext, b: fp.EFloatContext, c: fp.EFloatContext, F: int,
       rho: fp.EFloatContext, k: int, e_zero: int | None = None) -> Design:
    """A T-FDPA chain: a fused sum with `c` inside."""
    return Design(a, b, c, L, rho, k, F=F, e_zero=e_zero)


_GROUP_SCALES: list[tuple[str, fp.EFloatContext | fp.ExpContext]] = [
    ('nvfp4', fp.MX_E4M3), ('mxfp4', fp.MX_E8M0)]

CATALOG: dict[str, Design] = {
    'nv.volta.f16.f32': _t(4, fp.FP16, fp.FP16, fp.FP32, 23, RZ_FP32, 8),
    'nv.volta.f16.f16': _t(4, fp.FP16, fp.FP16, fp.FP16, 23, RNE_FP16, 8),
    'nv.turing.f16.f32': _t(8, fp.FP16, fp.FP16, fp.FP32, 24, RZ_FP32, 16),
    'nv.turing.f16.f16': _t(8, fp.FP16, fp.FP16, fp.FP16, 24, RNE_FP16, 16),
    'nv.ampere.tf32.f32': _t(4, fp.TF32, fp.TF32, fp.FP32, 24, RZ_FP32, 8),
    'nv.ampere.bf16.f32': _t(8, fp.BF16, fp.BF16, fp.FP32, 24, RZ_FP32, 16),
    **{f'nv.ada.{tag}.f32': _t(16, a, b, fp.FP32, 13, RZ_E8M13, 16, -132) for tag, a, b in _FP8_PAIRS},
    **{f'nv.ada.{tag}.f16': _t(16, a, b, fp.FP16, 13, RNE_FP16, 16, -21) for tag, a, b in _FP8_PAIRS},
    'nv.hopper.tf32.f32': _t(8, fp.TF32, fp.TF32, fp.FP32, 25, RZ_FP32, 16),
    'nv.hopper.bf16.f32': _t(16, fp.BF16, fp.BF16, fp.FP32, 25, RZ_FP32, 32),
    'nv.hopper.f16.f32': _t(16, fp.FP16, fp.FP16, fp.FP32, 25, RZ_FP32, 32),
    'nv.hopper.f16.f16.wgmma': _t(16, fp.FP16, fp.FP16, fp.FP16, 25, RNE_FP16, 32, -133),
    'nv.hopper.f16.f16.mma': _t(16, fp.FP16, fp.FP16, fp.FP16, 25, RNE_FP16, 32),
    **{f'nv.hopper.{tag}.f32': _t(32, a, b, fp.FP32, 13, RZ_E8M13, 32, -133) for tag, a, b in _FP8_PAIRS},
    **{f'nv.hopper.{tag}.f16': _t(32, a, b, fp.FP16, 13, RNE_FP16, 32, -133) for tag, a, b in _FP8_PAIRS},
    **{f'nv.blackwell.{tag}.f32': _t(32, a, b, fp.FP32, 25, RZ_FP32, 32) for tag, a, b in _F8F6F4_PAIRS},
    **{f'nv.blackwell.{tag}.f16.tcgen05': _t(32, a, b, fp.FP16, 25, RNE_FP16, 32, -133)
       for tag, a, b in _F8F6F4_PAIRS},
    **{f'nv.rtx_blackwell.{tag}.f16.mma': _t(32, a, b, fp.FP16, 25, RNE_FP16, 32)
       for tag, a, b in _F8F6F4_PAIRS},
    **{f'nv.blackwell.mx.{tag}': replace(_t(32, a, b, fp.FP32, 25, RZ_FP32, 32), scales=fp.MX_E8M0)
       for tag, a, b in _F8F6F4_PAIRS},
    **{f'nv.blackwell.{tag}': replace(_t(64, fp.MX_E2M1, fp.MX_E2M1, fp.FP32, 35, RZ_FP32, 64),
                                      scales=s, G=16)
       for tag, s in _GROUP_SCALES},
    'amd.cdna1.bf16': Design(fp.BF16, fp.BF16, fp.FP32, 2, fp.FP32, 4, sum='exact'),
    'amd.cdna1.f16': Design(fp.FP16, fp.FP16, fp.FP32, 4, fp.FP32, 4, sum='exact'),
    **{name: Design(t, t, fp.FP32, P, fp.FP32, 4, sum='tree', products='rounded', ftz=True)
       for name, t, P in [('amd.cdna2.bf16', fp.BF16, 2), ('amd.cdna2.bf16_1k', fp.BF16, 4),
                          ('amd.cdna2.f16', fp.FP16, 4)]},
    **{f'amd.cdna3.{tag}': Design(t, t, fp.FP32, L, fp.FP32, 8, join='separate')
       for tag, t, L in [('tf32', fp.TF32, 4), ('f16', fp.FP16, 8), ('bf16', fp.BF16, 8)]},
    **{f'amd.cdna3.{tag}': Design(a, b, fp.FP32, 16, fp.FP32, 16, join='separate', groups='even-odd')
       for tag, a, b in _CDNA3_FP8_PAIRS},
    'fp64 (fma)': Design(fp.FP64, fp.FP64, fp.FP64, 1, fp.FP64, 4, sum='fma'),
    'fp32 (fma)': Design(fp.FP32, fp.FP32, fp.FP32, 1, fp.FP32, 4, sum='fma'),
}
