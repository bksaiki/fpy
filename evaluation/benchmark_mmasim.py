"""Benchmark compiled FPy MMA fixed-length dot products against MMA-Sim.

MMA-Sim runs eagerly and, unless --no-compile is given, under torch.compile.
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
from operator import itemgetter
from pathlib import Path
from typing import Any

EVALUATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_ROOT.parent
DEFAULT_MMASIM_ROOT = Path.home() / "repos" / "mma-sim"
TORCH_EXTENSIONS_DIR = EVALUATION_ROOT / ".torch_extensions"

os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(TORCH_EXTENSIONS_DIR))

import torch
import torch._dynamo
from torch import Tensor
from torch.utils import benchmark

FPY_MODELS_ROOT = REPO_ROOT / "cpp_extension"
sys.path.insert(0, str(FPY_MODELS_ROOT))

try:
    import fpy2_models
    from fpy2_models.ops import _quantize_tf32
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
        "  cd cpp_extension && ../.venv/bin/python setup.py build_ext --inplace"
    ) from exc


TF32 = "tf32"
E2M1 = "e2m1"
"""Operand formats without a torch dtype. MMA-Sim takes TF32 operands as FP32,
which it truncates itself. It rejects packed E2M1 operands outside its
group-scaled model, so it takes their values as FP16, as that model does after
unpacking them."""

Format = torch.dtype | str


def mmasim_dtype(fmt: Format) -> torch.dtype:
    """The dtype of MMA-Sim's operands in format *fmt*."""
    return {TF32: torch.float32, E2M1: torch.float16}.get(fmt, fmt)


@dataclass(frozen=True)
class Case:
    """One pair of equivalent FPy and MMA-Sim arithmetic kernels."""

    name: str
    reference: object
    a_format: Format
    b_format: Format
    accumulator_dtype: torch.dtype
    k: int
    op_name: str
    compute_dtype: torch.dtype
    kind: str = "dpa"
    scale_block: int | None = None
    scale_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class Prepared:
    """Inputs and zero-argument callables for one timed configuration."""

    fpy: Callable[[], Tensor]
    mmasim: Callable[[], Tensor]
    mmasim_compile_naive: Callable[[], Tensor] | None = None


MANIFEST = FPY_MODELS_ROOT / "build" / "generated_models" / "manifest.json"


def _manifest() -> dict[str, Any]:
    """The models built into fpy2_models, written by generate_models.py."""
    return json.loads(MANIFEST.read_text())


def unsupported() -> list[dict[str, str]]:
    return [
        {"design": name, "reason": reason}
        for name, reason in _manifest()["unsupported"].items()
    ]


def _configure_macos_compiler() -> None:
    """Select the installed OpenMP-capable compiler needed by MMA-Sim FMA."""
    if platform.system() != "Darwin" or "CXX" in os.environ:
        return
    homebrew_clang = Path("/opt/homebrew/opt/llvm/bin/clang++")
    if homebrew_clang.exists():
        os.environ["CXX"] = str(homebrew_clang)


def build_cases(mmasim_root: Path) -> list[Case]:
    """Import the requested checkout and construct all matched cases."""
    package_root = mmasim_root / "src" / "mmasim"
    if not package_root.is_dir():
        raise SystemExit(
            f"MMA-Sim package not found at {package_root}. "
            "Pass its checkout with --mmasim-root."
        )
    sys.path.insert(0, str(package_root))
    _configure_macos_compiler()

    try:
        from mmasim import amd, nv_ptx
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "Could not import the local MMA-Sim simulators. On macOS, "
            "MMA-Sim's FP64 helper requires an OpenMP-capable compiler; install "
            "Homebrew LLVM or set CXX to one before running the benchmark.\n"
            f"Original error: {exc}"
        ) from exc

    # MMA-Sim's instruction simulators select each design's arithmetic model
    # and its parameters from the architecture and instruction type.
    specs = _references(nv_ptx, amd)
    cases = []
    for model in _manifest()["models"]:
        name = model["name"]
        if name not in specs:
            raise SystemExit(f"No MMA-Sim reference for FPy design {name}")
        reference, a_format, b_format, accumulator, *scaling = specs[name]
        scale_block, scale_dtype = scaling or (None, None)
        cases.append(
            Case(
                # `fp64 (fma)` -> `fp64.fma`
                name.replace(" (", ".").rstrip(")"),
                reference,
                a_format,
                b_format,
                accumulator,
                model["k"],
                model["op"],
                torch.float64 if model["compute"] == "double" else torch.float32,
                model["kind"],
                scale_block,
                scale_dtype,
            )
        )
    return cases


def _references(nv_ptx: Any, amd: Any) -> dict[str, tuple]:
    """Map each FPy design to (MMA-Sim simulator, A format, B format,
    accumulator dtype[, scale block, scale dtype])."""
    f16, bf16, f32, f64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
    e8m0, e4m3 = torch.float8_e8m0fnu, torch.float8_e4m3fn
    nv = {"e4m3": e4m3, "e5m2": torch.float8_e5m2, "e2m1": E2M1}
    cdna3 = {"fp8": torch.float8_e4m3fnuz, "bf8": torch.float8_e5m2fnuz}
    fp8_pairs = [("e4m3", "e4m3"), ("e5m2", "e5m2"), ("e4m3", "e5m2")]
    f8f6f4_pairs = fp8_pairs + [("e2m1", "e2m1"), ("e4m3", "e2m1"), ("e5m2", "e2m1")]

    def tag(a: str, b: str) -> str:
        return a if a == b else f"{a}.{b}"

    specs = {
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
            specs[f"nv.ada.{tag(a, b)}.{out}"] = (
                nv_ptx.MMA("Ada Lovelace", f"m16n8k32.{out}.{a}.{b}.{out}"), A, B, acc
            )
            specs[f"nv.hopper.{tag(a, b)}.{out}"] = (
                nv_ptx.WGMMA("Hopper", f"m64n8k32.{out}.{a}.{b}"), A, B, acc
            )
    for a, b in f8f6f4_pairs:
        A, B = nv[a], nv[b]
        specs[f"nv.blackwell.{tag(a, b)}.f32"] = (
            nv_ptx.TCGen05MMA("Blackwell", f"m128n8k32.f32.{a}.{b}"), A, B, f32
        )
        specs[f"nv.blackwell.{tag(a, b)}.f16.tcgen05"] = (
            nv_ptx.TCGen05MMA("Blackwell", f"m128n8k32.f16.{a}.{b}"), A, B, f16
        )
        specs[f"nv.rtx_blackwell.{tag(a, b)}.f16.mma"] = (
            nv_ptx.MMA("RTX Blackwell", f"m16n8k32.f16.{a}.{b}.f16"), A, B, f16
        )
        specs[f"nv.blackwell.mx.{tag(a, b)}"] = (
            nv_ptx.TCGen05MMABlockScale(
                "Blackwell", f"m128n8k32.block32.f32.{a}.{b}.ue8m0"
            ),
            A, B, f32, 32, e8m0,
        )
    for a, b in (("fp8", "fp8"), ("bf8", "bf8"), ("fp8", "bf8")):
        specs[f"amd.cdna3.{tag(a, b)}"] = (
            amd.MFMA("CDNA3", f"f32_32x32x16_{a}_{b}"), cdna3[a], cdna3[b], f32
        )
    return specs


def select_cases(cases: Sequence[Case], filters: Sequence[str]) -> list[Case]:
    """Select cases whose names contain any requested substring."""
    if not filters:
        return list(cases)
    selected = [case for case in cases if any(f in case.name for f in filters)]
    if not selected:
        choices = ", ".join(case.name for case in cases)
        raise SystemExit(f"No design matched {list(filters)!r}. Choices: {choices}")
    return selected


# Input distributions, sampled in FP64 before rounding to each design's format.
DISTRIBUTIONS = {
    "normal": "normal(0, 0.5)",
    "uniform": "uniform[-1, 1)",
}


def _source_values(
    batch: int, seed: int, length: int, distribution: str = "normal"
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, length)
    if distribution == "normal":
        return torch.randn(shape, dtype=torch.float64, generator=generator) * 0.5
    if distribution == "uniform":
        return torch.rand(shape, dtype=torch.float64, generator=generator) * 2 - 1
    raise ValueError(f"unknown distribution {distribution!r}")


def _operand_views(
    fmt: Format, source: Tensor, compute_dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Return (MMA-Sim storage view, FPy C++ compute view) in format *fmt*."""
    if fmt == TF32:
        storage = source.float()
        return storage, _quantize_tf32(storage)
    if fmt == E2M1:
        values = f4_unpacked_to_f32(f32_to_f4_unpacked(source.float().contiguous()))
        return values.half(), values
    storage = source.to(fmt)
    return storage, storage.to(compute_dtype)


def _block_scaled_operand(
    case: Case, fmt: Format, source: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (MMA-Sim values, FPy values, scales) from TorchAO block quantization."""
    source = source.float().contiguous()
    if case.scale_dtype == torch.float8_e4m3fn:
        # Single-level NVFP4: the per-tensor scale lies outside the MMA.
        scales, packed = nvfp4_quantize(source, case.scale_block)
    else:
        elem_dtype = torch.float4_e2m1fn_x2 if fmt == E2M1 else fmt
        scales, packed = to_mx(source, elem_dtype, case.scale_block)
    if fmt == E2M1:
        values = f4_unpacked_to_f32(unpack_uint4(packed.view(torch.uint8)))
        return values.half(), values, scales
    return packed, packed.float(), scales


def prepare_case(
    case: Case,
    batch: int,
    seed: int,
    length: int = 1024,
    compiled: bool = False,
    distribution: str = "normal",
) -> Prepared:
    """Prepare full operands and chunk views outside the timed region.

    Block-scaled designs carry one scale per block, sliced per model-sized
    chunk for MMA-Sim. All implementations start each invocation at zero and
    preserve the model's accumulator rounding per chunk. With `compiled`, also
    prepare unmodified MMA-Sim's `dpa` under default `torch.compile` settings,
    graph breaks included, called in the eager chunk loop; it compiles on its
    first call.
    """
    if length <= 0 or length % case.k:
        raise ValueError(f"length must be a positive multiple of {case.k}")
    a_source = _source_values(batch, seed, length, distribution)
    b_source = _source_values(batch, seed + 1, length, distribution)
    if case.kind == "dpa":
        a_mmasim, a_fpy = _operand_views(case.a_format, a_source, case.compute_dtype)
        b_mmasim, b_fpy = _operand_views(case.b_format, b_source, case.compute_dtype)
        operands = [(a_mmasim, case.k), (b_mmasim, case.k)]
        fpy_scales = ()
    elif case.kind in ("scaled", "group_scaled"):
        a_mmasim, a_fpy, a_scales = _block_scaled_operand(case, case.a_format, a_source)
        b_mmasim, b_fpy, b_scales = _block_scaled_operand(case, case.b_format, b_source)
        scales_per_chunk = case.k // case.scale_block
        operands = [
            (a_mmasim, case.k),
            (b_mmasim, case.k),
            (a_scales, scales_per_chunk),
            (b_scales, scales_per_chunk),
        ]
        fpy_scales = (a_scales.float(), b_scales.float())
    else:
        raise AssertionError(f"unknown case kind: {case.kind}")
    fpy_op = getattr(torch.ops.fpy2_models, case.op_name + "_dot").default
    # Contiguous chunk tensors avoid charging MMA-Sim for strided input copies.
    chunks = [
        tuple(
            tensor[:, i * width : (i + 1) * width].contiguous()
            for tensor, width in operands
        )
        for i in range(length // case.k)
    ]

    def mmasim_dot() -> Tensor:
        acc = torch.zeros((batch,), dtype=case.accumulator_dtype)
        for a_chunk, b_chunk, *scales in chunks:
            acc = case.reference.dpa(a_chunk, b_chunk, acc, *scales)
        return acc

    mmasim_compile_naive = None
    if compiled:
        naive_dpa = torch.compile(case.reference.dpa)

        def mmasim_compile_naive() -> Tensor:
            acc = torch.zeros((batch,), dtype=case.accumulator_dtype)
            for a_chunk, b_chunk, *scales in chunks:
                acc = naive_dpa(a_chunk, b_chunk, acc, *scales)
            return acc

    return Prepared(
        fpy=lambda: fpy_op(a_fpy, b_fpy, *fpy_scales),
        mmasim=mmasim_dot,
        mmasim_compile_naive=mmasim_compile_naive,
    )


def _integer_view(tensor: Tensor) -> Tensor:
    tensor = tensor.detach().contiguous()
    if tensor.dtype == torch.float16:
        return tensor.view(torch.int16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    if tensor.dtype == torch.float64:
        return tensor.view(torch.int64)
    raise TypeError(f"cannot compare bit patterns for {tensor.dtype}")


def _compare(actual: Tensor, expected: Tensor, implementation: str) -> dict[str, Any]:
    """Compare one implementation's output with eager MMA-Sim bit for bit."""
    prefix = {
        "fpy-core": "fpy",
        "mmasim-compile-naive": "naive",
    }[implementation]
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "valid": False,
            "implementation": implementation,
            "reason": "shape or dtype mismatch",
            f"{prefix}_shape": list(actual.shape),
            "mmasim_shape": list(expected.shape),
            f"{prefix}_dtype": str(actual.dtype),
            "mmasim_dtype": str(expected.dtype),
        }

    actual_bits = _integer_view(actual)
    expected_bits = _integer_view(expected)
    differences = (actual_bits != expected_bits).nonzero()
    if differences.numel() == 0:
        return {"valid": True}

    flat_index = int(differences[0].item())
    width = actual.element_size() * 8
    mask = (1 << width) - 1
    return {
        "valid": False,
        "implementation": implementation,
        "reason": "output bit mismatch",
        "first_difference": flat_index,
        f"{prefix}_value": float(actual.flatten()[flat_index]),
        "mmasim_value": float(expected.flatten()[flat_index]),
        f"{prefix}_bits": (
            f"0x{int(actual_bits.flatten()[flat_index]) & mask:0{width // 4}x}"
        ),
        "mmasim_bits": (
            f"0x{int(expected_bits.flatten()[flat_index]) & mask:0{width // 4}x}"
        ),
    }


def validate_naive(prepared: Prepared) -> dict[str, Any]:
    """Validate out-of-the-box torch.compile, recording failures as results.

    Default torch.compile can fail to compile MMA-Sim or miscompile it; either
    outcome excludes only this series from timing.
    """
    try:
        actual = prepared.mmasim_compile_naive()
    except Exception as exc:
        return {
            "valid": False,
            "implementation": "mmasim-compile-naive",
            "reason": "compile or runtime error",
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
        }
    return _compare(actual, prepared.mmasim(), "mmasim-compile-naive")


def _instruction(reference: object) -> str:
    """NVIDIA simulators name their instruction `shape_and_type`; AMD, `suffix`."""
    return getattr(reference, "shape_and_type", None) or reference.suffix


def _narrow(tensor: Tensor, dtype: torch.dtype) -> Tensor | None:
    """*tensor* in *dtype* if that is exact, else None.

    FPy's C++ models compute FP16-output designs in FP32 storage; their results
    are FP16 values.
    """
    narrowed = tensor.to(dtype)
    widened = narrowed.to(tensor.dtype)
    exact = (widened == tensor) | (widened.isnan() & tensor.isnan())
    return narrowed if bool(exact.all()) else None


def validate(prepared: Prepared) -> dict[str, Any]:
    """Return a serializable bitwise-validation result against eager MMA-Sim."""
    candidates = {"fpy-core": prepared.fpy}
    # No input requires grad. Validating in the timed grad mode avoids a
    # second torch.compile for the global-state change.
    expected = prepared.mmasim()
    for implementation, fn in candidates.items():
        actual = fn()
        if implementation == "fpy-core" and actual.dtype != expected.dtype:
            actual = _narrow(actual, expected.dtype)
            if actual is None:
                return {
                    "valid": False,
                    "implementation": implementation,
                    "reason": f"FPy output is not exactly representable in {expected.dtype}",
                }
        result = _compare(actual, expected, implementation)
        if not result["valid"]:
            return result
    return {"valid": True}


def _quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    median = statistics.median(values)
    if len(values) == 1:
        return median, median, 0.0
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return median, q1, q3 - q1


def time_callable(
    fn: Callable[[], Tensor],
    *,
    implementation: str,
    case: Case,
    batch: int,
    length: int,
    threads: int,
    replicate: int,
    min_run_time: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Collect normalized raw Timer blocks for one implementation."""
    rows: list[dict[str, Any]] = []
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn}, num_threads=threads)
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    samples = list(measurement.raw_times)
    rng.shuffle(samples)
    for sample, raw_seconds in enumerate(samples):
        seconds_per_call = raw_seconds / measurement.number_per_run
        rows.append(
            {
                "design": case.name,
                "implementation": implementation,
                "batch_size": batch,
                "dot_length": length,
                "model_width": case.k,
                "chunks_per_dot": length // case.k,
                "threads": threads,
                "replicate": replicate,
                "sample": sample,
                "number_per_run": measurement.number_per_run,
                "raw_seconds": raw_seconds,
                "seconds_per_call": seconds_per_call,
            }
        )
    return rows


CONFIG_FIELDS = (
    "design",
    "implementation",
    "batch_size",
    "dot_length",
    "model_width",
    "threads",
)
config_key = itemgetter(*CONFIG_FIELDS)
summary_order = itemgetter(
    "design", "batch_size", "dot_length", "model_width", "implementation"
)
METRIC_FIELDS = (
    "median_seconds_per_call",
    "q1_seconds_per_call",
    "iqr_seconds_per_call",
    "relative_iqr",
    "ns_per_dot",
    "dots_per_second",
    "terms_per_second",
)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[config_key(row)].append(row["seconds_per_call"])

    summaries = []
    for key, values in groups.items():
        config = dict(zip(CONFIG_FIELDS, key))
        batch, length, width = (
            config[k] for k in ("batch_size", "dot_length", "model_width")
        )
        median, q1, iqr = _quartiles(values)
        # Keep CSV columns in their established order.
        threads = config.pop("threads")
        summaries.append(
            {
                **config,
                "chunks_per_dot": length // width,
                "threads": threads,
                "samples": len(values),
                "median_seconds_per_call": median,
                "q1_seconds_per_call": q1,
                "iqr_seconds_per_call": iqr,
                "relative_iqr": iqr / median,
                "ns_per_dot": median * 1e9 / batch,
                "dots_per_second": batch / median,
                "terms_per_second": batch * length / median,
            }
        )
    return sorted(summaries, key=summary_order)


def enrich_measurements(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> None:
    """Attach configuration aggregates to every normalized raw sample."""
    by_key = {config_key(row): row for row in summaries}
    for row in rows:
        summary = by_key[config_key(row)]
        row.update((field, summary[field]) for field in METRIC_FIELDS)


def _run_command(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "revision": _run_command(["git", "rev-parse", "HEAD"], path),
        "branch": _run_command(["git", "branch", "--show-current"], path),
        "dirty": bool(_run_command(["git", "status", "--porcelain"], path)),
    }


def _hardware_metadata() -> dict[str, str]:
    """Return a non-identifying subset of the macOS hardware report."""
    report = _run_command(["system_profiler", "SPHardwareDataType"])
    if not report:
        return {}
    wanted = {
        "Model Name": "model_name",
        "Model Identifier": "model_identifier",
        "Chip": "chip",
        "Total Number of Cores": "cores",
        "Memory": "memory",
    }
    hardware = {}
    for line in report.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key in wanted:
            hardware[wanted[key]] = value.strip()
    return hardware


def collect_metadata(args: argparse.Namespace, mmasim_root: Path) -> dict[str, Any]:
    cxx = os.environ.get("CXX", "c++")
    hardware = _hardware_metadata()
    cpu = hardware.get("chip") or platform.processor() or platform.machine()
    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_version = None
    return {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "fpy": _git_metadata(REPO_ROOT),
        "mmasim": _git_metadata(mmasim_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchao": torchao_version,
        "platform": platform.platform(),
        "cpu": cpu,
        "hardware": hardware,
        "logical_cpu_count": os.cpu_count(),
        "torch_threads_before_benchmark": {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
        "benchmark_threads": args.threads,
        "compiler": {
            "CXX": cxx,
            "version": _run_command([cxx, "--version"]),
            "fpy_extension_flags": [
                "-O3",
                "-DPy_LIMITED_API=0x03090000",
                "-DTORCH_TARGET_VERSION=0x020a000000000000",
            ],
            "mmasim_fma_flags": ["-O3", "-fopenmp"],
            "mmasim_compile_naive": (
                "torch.compile(reference.dpa) with default settings on "
                "unmodified MMA-Sim; eager chunk loop"
                if args.compile
                else None
            ),
        },
        "parameters": {
            "design_filters": args.design,
            "batch_sizes": args.batch_sizes,
            "dot_lengths": args.dot_lengths,
            "initial_accumulator": 0,
            "distribution": args.distribution,
            "inputs": f"{DISTRIBUTIONS[args.distribution]}; "
            "MXFP8/NVFP4 block-quantized with TorchAO",
            "fpy_chunk_loop": "native",
            "seed": args.seed,
            "replicates": args.replicates,
            "min_run_time": args.min_run_time,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmasim-root", type=Path, default=DEFAULT_MMASIM_ROOT)
    parser.add_argument(
        "--design",
        action="append",
        default=[],
        help="design-name substring; repeat to select multiple designs",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("1,16,256"),
    )
    parser.add_argument(
        "--dot-lengths", type=parse_int_list, default=parse_int_list("1024,2048,8192")
    )
    parser.add_argument(
        "--distribution",
        choices=list(DISTRIBUTIONS),
        default="normal",
        help="input sampling: "
        + ", ".join(f"{name} = {text}" for name, text in DISTRIBUTIONS.items()),
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--min-run-time", type=float, default=0.25)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="skip the torch.compile MMA-Sim series",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.replicates <= 0 or args.min_run_time <= 0:
        parser.error("threads, replicates, and min-run-time must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    mmasim_root = args.mmasim_root.expanduser().resolve()
    cases = select_cases(build_cases(mmasim_root), args.design)

    for case in cases:
        if any(length % case.k for length in args.dot_lengths):
            raise SystemExit(
                f"dot lengths must be multiples of {case.k} for {case.name}"
            )

    if args.output is None:
        run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output_dir = EVALUATION_ROOT / "results" / run_id
    else:
        output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = collect_metadata(args, mmasim_root)
    write_json(output_dir / "metadata.json", metadata)

    validations: list[dict[str, Any]] = [
        {**entry, "status": "unsupported"} for entry in unsupported()
    ]
    rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    started = time.monotonic()

    configurations = product(enumerate(cases), args.dot_lengths, args.batch_sizes)
    for (case_index, case), length, batch in configurations:
        case_seed = args.seed + case_index * 1_000_003 + batch
        # Each configuration compiles from scratch, as in a fresh process, so
        # default recompile limits apply per configuration.
        torch._dynamo.reset()
        prepared = prepare_case(
            case, batch, case_seed, length, args.compile, args.distribution
        )
        result = validate(prepared)
        validation = {
            "design": case.name,
            "instruction": f"{case.reference.arch} {_instruction(case.reference)}",
            "batch_size": batch,
            "dot_length": length,
            "model_width": case.k,
            "seed": case_seed,
            **result,
        }
        if prepared.mmasim_compile_naive is not None:
            naive = validate_naive(prepared)
            validation["mmasim_compile_naive"] = naive
        validations.append(validation)
        label = f"{case.name} batch={batch} length={length}"
        if prepared.mmasim_compile_naive is not None and not naive["valid"]:
            print(
                f"NAIVE   {label}: {naive['reason']} {naive.get('error', '')}",
                flush=True,
            )
        if not result["valid"]:
            print(
                f"INVALID {label}: {result['reason']}",
                flush=True,
            )
            continue
        if args.validate_only:
            print(
                f"VALID   {label}",
                flush=True,
            )
            continue

        implementations = [
            ("fpy-core", prepared.fpy),
            ("mmasim", prepared.mmasim),
        ]
        if prepared.mmasim_compile_naive is not None and naive["valid"]:
            implementations.append(
                ("mmasim-compile-naive", prepared.mmasim_compile_naive)
            )
        for _, fn in implementations:
            fn()
        for replicate in range(args.replicates):
            rng.shuffle(implementations)
            for implementation, fn in implementations:
                rows.extend(
                    time_callable(
                        fn,
                        implementation=implementation,
                        case=case,
                        batch=batch,
                        length=length,
                        threads=args.threads,
                        replicate=replicate,
                        min_run_time=args.min_run_time,
                        rng=rng,
                    )
                )
        print(
            f"TIMED   {label}",
            flush=True,
        )

    write_json(output_dir / "validation.json", validations)
    summaries = summarize(rows)
    enrich_measurements(rows, summaries)
    write_csv(output_dir / "measurements.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["valid_configurations"] = sum(
        record.get("valid") is True for record in validations
    )
    metadata["invalid_configurations"] = sum(
        record.get("valid") is False for record in validations
    )
    # Out-of-the-box torch.compile failures are results, not benchmark errors.
    metadata["naive_compile_failures"] = sum(
        record.get("mmasim_compile_naive", {}).get("valid") is False
        for record in validations
    )
    write_json(output_dir / "metadata.json", metadata)
    print(f"\nResults: {output_dir}")
    return 1 if metadata["invalid_configurations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
