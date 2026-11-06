"""
Each evaluation's default configuration: what it measures unless told
otherwise.  Machine settings (processes, output paths) stay with each script.

Needs `common/mma` and `examples/mmasim` on the import path, as every
evaluation's entry point puts them.
"""

from dataclasses import dataclass, replace
from typing import Any

import generator
from generator.catalog import CATALOG

import fpy2 as fp

MODEL = 'Qwen/Qwen3.5-0.8B'
"""The LLM the LLM evaluations run (`dse/llm`, `gsm8k`); of `swap.MODELS`."""


# ---------------------------------------------------------------------------
# The design space (`dse/llm`, `dse/blas`)

GRID_ANCHORS = ('nv.hopper.bf16.f32', 'nv.ampere.bf16.f32', 'amd.cdna3.bf16', 'amd.cdna2.bf16',
                'nv.blackwell.mx.e4m3', 'nv.hopper.e4m3.f32', 'nv.ada.e4m3.f32', 'amd.cdna3.fp8')
"""The grid's anchors: each generation's BF16 and FP8 instruction (MMA-Sim's
Blackwell BF16 is Hopper's, Ada's Ampere's; CDNA2 has no FP8); see :func:`grid`."""

SCHEMES = {'nv.hopper.bf16.f32': 'bf16', 'nv.ampere.bf16.f32': 'bf16', 'amd.cdna3.bf16': 'bf16',
           'amd.cdna2.bf16': 'bf16', 'nv.blackwell.mx.e4m3': 'mxfp8', 'nv.hopper.e4m3.f32': 'fp8-row',
           'nv.ada.e4m3.f32': 'fp8-row', 'amd.cdna3.fp8': 'fp8-row:fnuz'}
"""Each anchor's quantization scheme, its designs' too."""

GRID_F = (10, 13, 16, 19, 22, 25, 28)
"""The `F` each anchor is swept over, its own added."""

SWAPPED_GUARD = CATALOG['amd.cdna3.bf16'].F2 - CATALOG['amd.cdna3.bf16'].F
"""A join made separate is CDNA3's: its adder this many bits wider than the sum."""


@dataclass(frozen=True)
class Point:
    anchor: str
    value: str
    """Its label: its `F`, its accumulator, and whether its join is swapped."""
    design: generator.Design

    @property
    def is_anchor(self) -> bool:
        return self.design == CATALOG[self.anchor]

    @property
    def about(self) -> dict[str, Any]:
        """Its design's parameters, readably (`generator.about` but its exact form)."""
        return {k: v for k, v in generator.about(self.design).items() if k != 'exact'}


def _reads_F(d: generator.Design) -> bool:
    """Whether *d*'s structure reads `F`: a fused sum or a separate join."""
    return d.sum == 'fused' or d.join == 'separate'


def _width(d: generator.Design, f: int) -> generator.Design:
    """*d* with `F` set to *f*; a separate join's `F2` with it, its guard bits kept."""
    return replace(d, F=f, **({'F2': f + d.F2 - d.F} if d.join == 'separate' else {}))


def _swap_join(d: generator.Design) -> generator.Design:
    """*d* with the other vendor's join: NVIDIA's `c` inside the fused sum,
    or CDNA3's added separately."""
    if d.join == 'inside':
        return replace(d, join='separate', F2=d.F + SWAPPED_GUARD)
    return replace(d, join='inside')


def grid() -> list[Point]:
    """Each anchor over `F`, and with its join swapped (`join=swapped` in the
    label); its accumulator its own.  A design that does not read `F`
    (CDNA2's tree) is its own alone."""
    out = []
    for a in GRID_ANCHORS:
        rho = CATALOG[a].rho
        acc = f'rho_bits={rho.nbits - 1 - rho.es} rho_mode={rho.rm.name}'
        for base, tail in ((CATALOG[a], ''), (_swap_join(CATALOG[a]), ' join=swapped')):
            mine = sorted({*GRID_F, base.F}) if _reads_F(base) else [base.F]
            out += [Point(a, f'F={f} {acc}{tail}', _width(base, f)) for f in mine]
    return out


# ---------------------------------------------------------------------------
# `dse/llm`

LLM_SEGMENTS = 10
"""WikiText-2 segments scored."""

LLM_SEED = 0
"""The segments' and random rounding's seed."""


# ---------------------------------------------------------------------------
# `dse/blas`

BLAS_SEED = 0
BLAS_N = 1024
"""Matrix size of GtB's tests."""

BLAS_FORMATS: dict[str, tuple[fp.Context, list[int]]] = {
    # name: (context, s)
    'BF16': (fp.BF16, [0, 11, 23, 34, 46, 57]),
    'E4M3': (fp.MX_E4M3, list(range(7))),
    # FNUZ tops out at 240, below 6 * 2^6
    'E4M3FNUZ': (fp.S1E4M3, list(range(6))),
}
"""Input formats, each with its scales `s`."""


# ---------------------------------------------------------------------------
# `analysis`

ANALYSIS_INPUTS_PER_DESIGN = 5000
ANALYSIS_SEED = 0
ANALYSIS_INPUTS = 'mixed'
"""Input source (`inputs.SOURCES`): random, every fifth adversarial."""


# ---------------------------------------------------------------------------
# `gsm8k`

GSM8K_DESIGNS = ('nv.hopper.bf16.f32',)
GSM8K_PROBLEMS = 32
"""Test problems."""

GSM8K_MAX_NEW = 256
"""Tokens per reply at most."""

GSM8K_BATCH = 32
"""Problems per batch."""

GSM8K_SAMPLE_STEPS = (4, 16)
"""Decode steps MMA-Sim's sample times: at least, at most."""

GSM8K_SAMPLE_CI = 0.01
"""The sample stops adding decode steps once their mean's 95% confidence
interval is within this fraction of it."""
