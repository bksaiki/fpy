"""Benchmark FPy's compiled MMA models against MMA-Sim.

For every design, dot length and batch size, the benchmark:

1. samples random inputs and rounds them to the design's formats;
2. checks that FPy's output matches MMA-Sim's bit for bit;
3. times both with PyTorch's `blocked_autorange`, in several replicates.

FPy's models and MMA-Sim's both model one K-wide instruction for a batch of
rows, with the same interface: `dpa(a, b, c)` with operands of shape (batch, K)
and accumulators of shape (batch,). Both compute dot products of any length the
same way: a Python loop calls `dpa` once per K-wide chunk, passing the
accumulators from chunk to chunk. With --compile, MMA-Sim's model is also
validated and timed under default `torch.compile`.

Each run writes a new results directory containing metadata.json,
validation.json, measurements.csv (one row per timed block) and summary.csv
(one row per design, implementation, length and batch size).
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

EVALUATION_ROOT = Path(__file__).resolve().parents[1]  # this file is in evaluation/performance
REPO_ROOT = EVALUATION_ROOT.parent
FPY_MODELS_ROOT = EVALUATION_ROOT / "common" / "fpy2_models"
MANIFEST = FPY_MODELS_ROOT / "build" / "generated_models" / "manifest.json"
DEFAULT_MMASIM_ROOT = Path.home() / "repos" / "mma-sim"

# MMA-Sim builds a C++ helper with torch's extension loader; keep it local.
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(EVALUATION_ROOT / ".torch_extensions"))

import torch
import torch._dynamo
from torch import Tensor
from torch.utils import benchmark

sys.path.insert(0, str(FPY_MODELS_ROOT))
try:
    import fpy2_models  # registers torch.ops.fpy2_models

    # The package lives in a directory of the same name. If only that outer
    # directory is importable, Python finds it as an empty namespace package
    # with no __file__ instead of failing.
    if fpy2_models.__file__ is None:
        raise ImportError(f"found {list(fpy2_models.__path__)} instead of the built package")
    from torchao.prototype.mx_formats.kernels import (
        f4_unpacked_to_f32,
        f32_to_f4_unpacked,
        unpack_uint4,
    )
    from torchao.prototype.mx_formats.mx_tensor import to_mx
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
except ImportError as exc:
    raise SystemExit(
        "Could not import the fpy2_models extension. Build it first with:\n"
        "  cd evaluation/common/fpy2_models && "
        "../../../.venv/bin/python setup.py build_ext --inplace\n"
        f"Original error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Designs: each FPy model paired with the MMA-Sim instruction it reproduces.

# Operand formats without a torch dtype. MMA-Sim takes TF32 operands as FP32
# and truncates them itself. It accepts E2M1 (FP4) operands only as FP16
# values, which is what its group-scaled model unpacks them to.
TF32 = "tf32"
E2M1 = "e2m1"
Format = torch.dtype | str


@dataclass(frozen=True)
class Case:
    """One FPy model and the MMA-Sim simulator it is compared with."""

    name: str
    reference: Any  # MMA-Sim simulator; `reference.dpa` computes one chunk
    a_format: Format
    b_format: Format
    accumulator_dtype: torch.dtype
    k: int  # products per model call
    op_name: str  # torch.ops.fpy2_models.<op_name> computes one chunk, like `dpa`
    compute_dtype: torch.dtype  # FPy's C++ operand type
    scale_block: int | None = None
    scale_dtype: torch.dtype | None = None


def manifest() -> dict[str, Any]:
    """The models built into fpy2_models, as listed by generate_models.py."""
    return json.loads(MANIFEST.read_text())


def instruction(reference: Any) -> str:
    """The instruction a simulator models: `shape_and_type` (NVIDIA) or `suffix` (AMD)."""
    return getattr(reference, "shape_and_type", None) or reference.suffix


def build_cases(mmasim_root: Path) -> list[Case]:
    """Pair every FPy model in the manifest with its MMA-Sim simulator."""
    package_root = mmasim_root / "src" / "mmasim"
    if not package_root.is_dir():
        raise SystemExit(
            f"MMA-Sim package not found at {package_root}. "
            "Pass its checkout with --mmasim-root."
        )
    sys.path.insert(0, str(package_root))
    # MMA-Sim's FMA helper needs OpenMP, which Apple's clang lacks.
    homebrew_clang = Path("/opt/homebrew/opt/llvm/bin/clang++")
    if platform.system() == "Darwin" and "CXX" not in os.environ and homebrew_clang.exists():
        os.environ["CXX"] = str(homebrew_clang)
    try:
        from mmasim import amd, nv_ptx
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "Could not import MMA-Sim. On macOS, its FMA helper needs an "
            "OpenMP-capable compiler: install Homebrew LLVM or set CXX.\n"
            f"Original error: {exc}"
        ) from exc

    references = mmasim_references(nv_ptx, amd)
    cases = []
    for model in manifest()["models"]:
        if model["name"] not in references:
            raise SystemExit(f"No MMA-Sim reference for FPy design {model['name']}")
        reference, a_format, b_format, accumulator, *scaling = references[model["name"]]
        scale_block, scale_dtype = scaling or (None, None)
        cases.append(
            Case(
                name=model["name"].replace(" (", ".").rstrip(")"),  # `fp64 (fma)` -> `fp64.fma`
                reference=reference,
                a_format=a_format,
                b_format=b_format,
                accumulator_dtype=accumulator,
                k=model["k"],
                op_name=model["op"],
                compute_dtype=torch.float64 if model["compute"] == "double" else torch.float32,
                scale_block=scale_block,
                scale_dtype=scale_dtype,
            )
        )
    return cases


def mmasim_references(nv_ptx: Any, amd: Any) -> dict[str, tuple]:
    """Map each FPy design name to (MMA-Sim simulator, A format, B format,
    accumulator dtype[, scale block, scale dtype])."""
    f16, bf16, f32, f64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
    e8m0, e4m3 = torch.float8_e8m0fnu, torch.float8_e4m3fn
    nv = {"e4m3": e4m3, "e5m2": torch.float8_e5m2, "e2m1": E2M1}
    cdna3 = {"fp8": torch.float8_e4m3fnuz, "bf8": torch.float8_e5m2fnuz}
    fp8_pairs = [("e4m3", "e4m3"), ("e5m2", "e5m2"), ("e4m3", "e5m2")]
    f8f6f4_pairs = fp8_pairs + [("e2m1", "e2m1"), ("e4m3", "e2m1"), ("e5m2", "e2m1")]

    def tag(a: str, b: str) -> str:
        return a if a == b else f"{a}.{b}"

    references = {
        "nv.volta.f16.f32": (nv_ptx.MMA("Volta", "m8n8k4.f32.f16.f16.f32"), f16, f16, f32),
        "nv.volta.f16.f16": (nv_ptx.MMA("Volta", "m8n8k4.f16.f16.f16.f16"), f16, f16, f16),
        "nv.turing.f16.f32": (nv_ptx.MMA("Turing", "m16n8k8.f32.f16.f16.f32"), f16, f16, f32),
        "nv.turing.f16.f16": (nv_ptx.MMA("Turing", "m16n8k8.f16.f16.f16.f16"), f16, f16, f16),
        "nv.ampere.tf32.f32": (
            nv_ptx.MMA("Ampere", "m16n8k8.f32.tf32.tf32.f32"), TF32, TF32, f32
        ),
        "nv.ampere.bf16.f32": (
            nv_ptx.MMA("Ampere", "m16n8k16.f32.bf16.bf16.f32"), bf16, bf16, f32
        ),
        "nv.hopper.tf32.f32": (nv_ptx.WGMMA("Hopper", "m64n8k8.f32.tf32.tf32"), TF32, TF32, f32),
        "nv.hopper.bf16.f32": (nv_ptx.WGMMA("Hopper", "m64n8k16.f32.bf16.bf16"), bf16, bf16, f32),
        "nv.hopper.f16.f32": (nv_ptx.WGMMA("Hopper", "m64n8k16.f32.f16.f16"), f16, f16, f32),
        "nv.hopper.f16.f16.wgmma": (nv_ptx.WGMMA("Hopper", "m64n8k16.f16.f16.f16"), f16, f16, f16),
        "nv.hopper.f16.f16.mma": (
            nv_ptx.MMA("Hopper", "m16n8k16.f16.f16.f16.f16"), f16, f16, f16
        ),
        "nv.blackwell.nvfp4": (
            nv_ptx.TCGen05MMABlockScale("Blackwell", "m128n8k64.block16.f32.e2m1.e2m1.ue4m3"),
            E2M1, E2M1, f32, 16, e4m3,
        ),
        "nv.blackwell.mxfp4": (
            nv_ptx.TCGen05MMABlockScale("Blackwell", "m128n8k64.block16.f32.e2m1.e2m1.ue8m0"),
            E2M1, E2M1, f32, 16, e8m0,
        ),
        "amd.cdna2.bf16": (amd.MFMA("CDNA2", "f32_32x32x4bf16"), bf16, bf16, f32),
        "amd.cdna2.bf16_1k": (amd.MFMA("CDNA2", "f32_32x32x8bf16_1k"), bf16, bf16, f32),
        "amd.cdna2.f16": (amd.MFMA("CDNA2", "f32_32x32x8f16"), f16, f16, f32),
        "amd.cdna3.tf32": (amd.MFMA("CDNA3", "f32_32x32x4_xf32"), TF32, TF32, f32),
        "amd.cdna3.f16": (amd.MFMA("CDNA3", "f32_32x32x8_f16"), f16, f16, f32),
        "amd.cdna3.bf16": (amd.MFMA("CDNA3", "f32_32x32x8_bf16"), bf16, bf16, f32),
        "fp64 (fma)": (nv_ptx.MMA("Ampere", "m8n8k4.f64.f64.f64.f64"), f64, f64, f64),
        "fp32 (fma)": (amd.MFMA("CDNA3", "f32_32x32x2_f32"), f32, f32, f32),
    }
    for a, b in fp8_pairs:
        A, B = nv[a], nv[b]
        for out, acc in (("f32", f32), ("f16", f16)):
            references[f"nv.ada.{tag(a, b)}.{out}"] = (
                nv_ptx.MMA("Ada Lovelace", f"m16n8k32.{out}.{a}.{b}.{out}"), A, B, acc
            )
            references[f"nv.hopper.{tag(a, b)}.{out}"] = (
                nv_ptx.WGMMA("Hopper", f"m64n8k32.{out}.{a}.{b}"), A, B, acc
            )
    for a, b in f8f6f4_pairs:
        A, B = nv[a], nv[b]
        references[f"nv.blackwell.{tag(a, b)}.f32"] = (
            nv_ptx.TCGen05MMA("Blackwell", f"m128n8k32.f32.{a}.{b}"), A, B, f32
        )
        references[f"nv.blackwell.{tag(a, b)}.f16.tcgen05"] = (
            nv_ptx.TCGen05MMA("Blackwell", f"m128n8k32.f16.{a}.{b}"), A, B, f16
        )
        references[f"nv.rtx_blackwell.{tag(a, b)}.f16.mma"] = (
            nv_ptx.MMA("RTX Blackwell", f"m16n8k32.f16.{a}.{b}.f16"), A, B, f16
        )
        references[f"nv.blackwell.mx.{tag(a, b)}"] = (
            nv_ptx.TCGen05MMABlockScale(
                "Blackwell", f"m128n8k32.block32.f32.{a}.{b}.ue8m0"
            ),
            A, B, f32, 32, e8m0,
        )
    for a, b in (("fp8", "fp8"), ("bf8", "bf8"), ("fp8", "bf8")):
        references[f"amd.cdna3.{tag(a, b)}"] = (
            amd.MFMA("CDNA3", f"f32_32x32x16_{a}_{b}"), cdna3[a], cdna3[b], f32
        )
    return references


def select_cases(cases: Sequence[Case], filters: Sequence[str]) -> list[Case]:
    """The cases whose names contain any of *filters*, or all cases."""
    if not filters:
        return list(cases)
    selected = [case for case in cases if any(f in case.name for f in filters)]
    if not selected:
        choices = ", ".join(case.name for case in cases)
        raise SystemExit(f"No design matched {list(filters)!r}. Choices: {choices}")
    return selected


# ---------------------------------------------------------------------------
# Inputs: the same values in MMA-Sim's and FPy's representations.

DISTRIBUTIONS = {"normal": "normal(0, 0.5)", "uniform": "uniform[-1, 1)"}


def sample(batch: int, length: int, seed: int, distribution: str) -> Tensor:
    """FP64 values to round to a design's input format."""
    generator = torch.Generator().manual_seed(seed)
    if distribution == "normal":
        return torch.randn((batch, length), dtype=torch.float64, generator=generator) * 0.5
    if distribution == "uniform":
        return torch.rand((batch, length), dtype=torch.float64, generator=generator) * 2 - 1
    raise ValueError(f"unknown distribution {distribution!r}")


def round_operand(fmt: Format, values: Tensor, compute_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """*values* rounded to *fmt*, as (MMA-Sim operand, FPy operand)."""
    if fmt == TF32:
        # Torch has no TF32 dtype. MMA-Sim's TF32 models truncate FP32 operands
        # with this helper; FPy's models expect operands already in TF32.
        from mmasim.arithmetic.helper import truncate_fp32_to_tf32

        mmasim = values.float()
        return mmasim, truncate_fp32_to_tf32(mmasim)
    if fmt == E2M1:
        fpy = f4_unpacked_to_f32(f32_to_f4_unpacked(values.float().contiguous()))
        return fpy.half(), fpy
    mmasim = values.to(fmt)
    return mmasim, mmasim.to(compute_dtype)


def quantize_blocks(case: Case, fmt: Format, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """*values* block-quantized with TorchAO, as (MMA-Sim operand, FPy operand, scales)."""
    values = values.float().contiguous()
    if case.scale_dtype == torch.float8_e4m3fn:
        # Single-level NVFP4: any per-tensor scale lies outside the MMA.
        scales, packed = nvfp4_quantize(values, case.scale_block)
    else:
        element_dtype = torch.float4_e2m1fn_x2 if fmt == E2M1 else fmt
        scales, packed = to_mx(values, element_dtype, case.scale_block)
    if fmt == E2M1:
        fpy = f4_unpacked_to_f32(unpack_uint4(packed.view(torch.uint8)))
        return fpy.half(), fpy, scales
    return packed, packed.float(), scales


@dataclass(frozen=True)
class Prepared:
    """Zero-argument functions that each compute the batch of dot products."""

    fpy: Callable[[], Tensor]
    mmasim: Callable[[], Tensor]
    mmasim_compiled: Callable[[], Tensor] | None = None


def prepare_case(
    case: Case,
    batch: int,
    seed: int,
    length: int = 1024,
    compiled: bool = False,
    distribution: str = "normal",
) -> Prepared:
    """Build the inputs for *batch* dot products of *length* terms, and the
    functions that compute them. Everything here happens outside the timing.

    Every dot product starts from a zero accumulator. With *compiled*, also
    wrap MMA-Sim's unmodified model in default `torch.compile`; it compiles on
    its first call.
    """
    if length <= 0 or length % case.k:
        raise ValueError(f"length must be a positive multiple of {case.k}")
    a_values = sample(batch, length, seed, distribution)
    b_values = sample(batch, length, seed + 1, distribution)
    if case.scale_block is None:
        a_mmasim, a_fpy = round_operand(case.a_format, a_values, case.compute_dtype)
        b_mmasim, b_fpy = round_operand(case.b_format, b_values, case.compute_dtype)
        scales = []
    else:
        a_mmasim, a_fpy, a_scales = quantize_blocks(case, case.a_format, a_values)
        b_mmasim, b_fpy, b_scales = quantize_blocks(case, case.b_format, b_values)
        scales = [a_scales, b_scales]

    fpy_scales = [s.to(case.compute_dtype) for s in scales]

    def chain(
        dpa: Callable[..., Tensor], accumulator: torch.dtype, *operands: Tensor
    ) -> Callable[[], Tensor]:
        """Dot products from one *dpa* call per K-wide chunk, passing the
        accumulator from chunk to chunk."""
        # Slice every operand, and its scales, into contiguous chunks ahead of
        # time, so timing excludes strided copies.
        chunks = list(
            zip(
                *(
                    [chunk.contiguous() for chunk in operand.chunk(length // case.k, dim=1)]
                    for operand in operands
                )
            )
        )

        def dot() -> Tensor:
            acc = torch.zeros(batch, dtype=accumulator)
            for a_chunk, b_chunk, *chunk_scales in chunks:
                acc = dpa(a_chunk, b_chunk, acc, *chunk_scales)
            return acc

        return dot

    fpy_dpa = getattr(torch.ops.fpy2_models, case.op_name).default
    mmasim_operands = (case.accumulator_dtype, a_mmasim, b_mmasim, *scales)
    return Prepared(
        fpy=chain(fpy_dpa, case.compute_dtype, a_fpy, b_fpy, *fpy_scales),
        mmasim=chain(case.reference.dpa, *mmasim_operands),
        mmasim_compiled=(
            chain(torch.compile(case.reference.dpa), *mmasim_operands) if compiled else None
        ),
    )


# ---------------------------------------------------------------------------
# Validation: outputs must match MMA-Sim's bit for bit.

BITS = {torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}


def compare(actual: Tensor, expected: Tensor, name: str) -> dict[str, Any]:
    """Compare *actual* with MMA-Sim's *expected* output bit for bit."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "valid": False,
            "reason": "shape or dtype mismatch",
            f"{name}_shape": list(actual.shape),
            "mmasim_shape": list(expected.shape),
            f"{name}_dtype": str(actual.dtype),
            "mmasim_dtype": str(expected.dtype),
        }
    actual_bits = actual.contiguous().view(BITS[actual.dtype]).flatten()
    expected_bits = expected.contiguous().view(BITS[expected.dtype]).flatten()
    differences = (actual_bits != expected_bits).nonzero()
    if differences.numel() == 0:
        return {"valid": True}

    index = int(differences[0])
    width = actual.element_size() * 8

    def hex_bits(bits: Tensor) -> str:
        return f"0x{int(bits[index]) & ((1 << width) - 1):0{width // 4}x}"

    return {
        "valid": False,
        "reason": "output bit mismatch",
        "first_difference": index,
        f"{name}_value": float(actual.flatten()[index]),
        "mmasim_value": float(expected.flatten()[index]),
        f"{name}_bits": hex_bits(actual_bits),
        "mmasim_bits": hex_bits(expected_bits),
    }


def validate(prepared: Prepared) -> dict[str, Any]:
    """Check FPy's output against eager MMA-Sim's."""
    expected = prepared.mmasim()
    actual = prepared.fpy()
    if actual.dtype != expected.dtype:
        # FPy computes FP16-output designs in FP32; their results must still be
        # exact FP16 values.
        narrowed = actual.to(expected.dtype)
        widened = narrowed.to(actual.dtype)
        if not bool(((widened == actual) | (widened.isnan() & actual.isnan())).all()):
            return {"valid": False, "reason": f"FPy output is not exactly {expected.dtype}"}
        actual = narrowed
    return compare(actual, expected, "fpy")


def validate_compiled(prepared: Prepared) -> dict[str, Any]:
    """Check torch.compile'd MMA-Sim against eager MMA-Sim. A failure to compile
    is a result, not an error."""
    try:
        actual = prepared.mmasim_compiled()
    except Exception as exc:  # noqa: BLE001 -- a compile failure is a result
        return {
            "valid": False,
            "reason": "compile or runtime error",
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
        }
    return compare(actual, prepared.mmasim(), "compiled")


# ---------------------------------------------------------------------------
# Timing and summary.


def time_blocks(fn: Callable[[], Tensor], threads: int, min_run_time: float) -> list[float]:
    """Seconds per call of *fn*, one value per block timed by `blocked_autorange`.

    `blocked_autorange` picks a number of calls per block that makes timer
    overhead negligible, then times blocks until *min_run_time* has passed.
    """
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn}, num_threads=threads)
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    return [seconds / measurement.number_per_run for seconds in measurement.raw_times]


SUMMARY_KEY = ("design", "implementation", "batch_size", "dot_length", "threads")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median and spread of the time per call, pooled over all blocks and
    replicates of each configuration."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in SUMMARY_KEY)].append(row["seconds_per_call"])

    summaries = []
    for key, times in sorted(groups.items()):
        config = dict(zip(SUMMARY_KEY, key))
        median = statistics.median(times)
        if len(times) > 1:
            q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
        else:
            q1 = q3 = median
        summaries.append(
            {
                **config,
                "samples": len(times),
                "median_seconds_per_call": median,
                "iqr_seconds_per_call": q3 - q1,
                "relative_iqr": (q3 - q1) / median,
                "ns_per_dot": median * 1e9 / config["batch_size"],
                "dots_per_second": config["batch_size"] / median,
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Output files.


def run(command: Sequence[str], cwd: Path | None = None) -> str | None:
    """The stripped output of *command*, or None if it fails."""
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "revision": run(["git", "rev-parse", "HEAD"], path),
        "dirty": bool(run(["git", "status", "--porcelain"], path)),
    }


def hardware_metadata() -> dict[str, str]:
    """Model, chip, cores and memory from the macOS hardware report, if available."""
    wanted = {"Model Name", "Chip", "Total Number of Cores", "Memory"}
    hardware = {}
    for line in (run(["system_profiler", "SPHardwareDataType"]) or "").splitlines():
        key, _, value = line.strip().partition(":")
        if key in wanted:
            hardware[key] = value.strip()
    return hardware or {"Chip": platform.processor() or platform.machine()}


def collect_metadata(args: argparse.Namespace, mmasim_root: Path) -> dict[str, Any]:
    cxx = os.environ.get("CXX", "c++")
    return {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "parameters": vars(args),
        "fpy": git_metadata(REPO_ROOT),
        "mmasim": git_metadata(mmasim_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchao": importlib.metadata.version("torchao"),
        "platform": platform.platform(),
        "hardware": hardware_metadata(),
        "compiler": {"CXX": cxx, "version": run([cxx, "--version"])},
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Command line.


def int_list(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mmasim-root", type=Path, default=DEFAULT_MMASIM_ROOT)
    parser.add_argument(
        "--design",
        action="append",
        default=[],
        help="design-name substring; repeat to select several designs",
    )
    parser.add_argument("--batch-sizes", type=int_list, default=[1024, 4096, 16384])
    parser.add_argument("--dot-lengths", type=int_list, default=[256, 1024, 4096])
    parser.add_argument(
        "--distribution",
        choices=list(DISTRIBUTIONS),
        default="normal",
        help="input distribution: "
        + ", ".join(f"{name} = {text}" for name, text in DISTRIBUTIONS.items()),
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--min-run-time", type=float, default=0.25)
    parser.add_argument("--validate-only", action="store_true", help="skip timing")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="also validate and time MMA-Sim under default torch.compile",
    )
    parser.add_argument("--output", type=Path, help="results directory to create")
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.replicates <= 0 or args.min_run_time <= 0:
        parser.error("threads, replicates and min-run-time must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Limit PyTorch and MMA-Sim's OpenMP helper to the requested threads.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    mmasim_root = args.mmasim_root.expanduser().resolve()
    cases = select_cases(build_cases(mmasim_root), args.design)
    for case in cases:
        if any(length % case.k for length in args.dot_lengths):
            raise SystemExit(f"dot lengths must be multiples of {case.k} for {case.name}")

    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output = (args.output or EVALUATION_ROOT / "results" / run_id).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = collect_metadata(args, mmasim_root)
    write_json(output / "metadata.json", metadata)
    started = time.monotonic()

    validations = [
        {"design": name, "status": "unsupported", "reason": reason}
        for name, reason in manifest()["unsupported"].items()
    ]
    rows = []
    rng = random.Random(args.seed)
    for (index, case), length, batch in product(
        enumerate(cases), args.dot_lengths, args.batch_sizes
    ):
        label = f"{case.name} batch={batch} length={length}"
        seed = args.seed + index * 1_000_003 + batch
        # Compile from scratch for every configuration, as a fresh process would.
        torch._dynamo.reset()
        prepared = prepare_case(case, batch, seed, length, args.compile, args.distribution)

        validation = {
            "design": case.name,
            "instruction": f"{case.reference.arch} {instruction(case.reference)}",
            "batch_size": batch,
            "dot_length": length,
            "seed": seed,
            **validate(prepared),
        }
        implementations = {"fpy-core": prepared.fpy, "mmasim": prepared.mmasim}
        if args.compile:
            compiled = validate_compiled(prepared)
            validation["mmasim_compiled"] = compiled
            if compiled["valid"]:
                implementations["mmasim-compiled"] = prepared.mmasim_compiled
            else:
                print(f"COMPILE {label}: {compiled['reason']} {compiled.get('error', '')}", flush=True)
        validations.append(validation)
        if not validation["valid"]:
            print(f"INVALID {label}: {validation['reason']}", flush=True)
            continue
        if args.validate_only:
            print(f"VALID   {label}", flush=True)
            continue

        # Warm up, then time every implementation once per replicate, in a
        # shuffled order so none is always measured first.
        for fn in implementations.values():
            fn()
        order = list(implementations.items())
        for replicate in range(args.replicates):
            rng.shuffle(order)
            for implementation, fn in order:
                for seconds in time_blocks(fn, args.threads, args.min_run_time):
                    rows.append(
                        {
                            "design": case.name,
                            "implementation": implementation,
                            "batch_size": batch,
                            "dot_length": length,
                            "threads": args.threads,
                            "replicate": replicate,
                            "seconds_per_call": seconds,
                        }
                    )
        print(f"TIMED   {label}", flush=True)

    write_json(output / "validation.json", validations)
    write_csv(output / "measurements.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    invalid = sum(record.get("valid") is False for record in validations)
    metadata.update(
        finished_at=datetime.now().astimezone().isoformat(),
        elapsed_seconds=time.monotonic() - started,
        valid_configurations=sum(record.get("valid") is True for record in validations),
        invalid_configurations=invalid,
        compile_failures=sum(
            record.get("mmasim_compiled", {}).get("valid") is False
            for record in validations
        ),
    )
    write_json(output / "metadata.json", metadata)
    print(f"\nResults: {output}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
