"""Benchmark compiled FPy MMA fixed-length dot products against eager MMA-Sim."""

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
from torch import Tensor
from torch.utils import benchmark

FPY_MODELS_ROOT = REPO_ROOT / "cpp_extension"
sys.path.insert(0, str(FPY_MODELS_ROOT))

try:
    import fpy2_models
    from fpy2_models.ops import _quantize_tf32
    from torchao.prototype.mx_formats.kernels import (
        f4_unpacked_to_f32,
        unpack_uint4,
    )
    from torchao.prototype.mx_formats.mx_tensor import to_mx
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
except ImportError as exc:
    raise SystemExit(
        "Could not import the fpy2_models extension. Build it first with:\n"
        "  cd cpp_extension && ../.venv/bin/python setup.py build_ext --inplace"
    ) from exc


@dataclass(frozen=True)
class Case:
    """One pair of equivalent FPy and MMA-Sim arithmetic kernels."""

    name: str
    k: int
    storage_dtype: torch.dtype
    reference: object
    kind: str = "dpa"
    tf32: bool = False
    scale_block: int | None = None

    @property
    def op_name(self) -> str:
        return self.name.replace(".", "_")

    @property
    def accumulator_dtype(self) -> torch.dtype:
        return torch.float64 if self.storage_dtype == torch.float64 else torch.float32


@dataclass(frozen=True)
class Prepared:
    """Inputs and zero-argument callables for one timed configuration."""

    fpy: Callable[[], Tensor]
    mmasim: Callable[[], Tensor]


UNSUPPORTED = (
    {
        "design": "amd.cdna1.bf16",
        "reason": "The FPy C++ backend cannot select storage for e_fdpa_block.",
    },
    {
        "design": "amd.cdna1.f16",
        "reason": "The FPy C++ backend cannot select storage for e_fdpa_block.",
    },
)


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
    return [
        Case(
            "nv.volta.f16.f32",
            8,
            torch.float16,
            nv_ptx.MMA("Volta", "m8n8k4.f32.f16.f16.f32"),
        ),
        Case(
            "nv.turing.f16.f32",
            16,
            torch.float16,
            nv_ptx.MMA("Turing", "m16n8k8.f32.f16.f16.f32"),
        ),
        Case(
            "nv.ampere.tf32.f32",
            8,
            torch.float32,
            nv_ptx.MMA("Ampere", "m16n8k8.f32.tf32.tf32.f32"),
            tf32=True,
        ),
        Case(
            "nv.ampere.bf16.f32",
            16,
            torch.bfloat16,
            nv_ptx.MMA("Ampere", "m16n8k16.f32.bf16.bf16.f32"),
        ),
        Case(
            "nv.ada.e5m2.f32",
            16,
            torch.float8_e5m2,
            nv_ptx.MMA("Ada Lovelace", "m16n8k16.f32.e5m2.e5m2.f32"),
        ),
        Case(
            "nv.hopper.f16.f32",
            32,
            torch.float16,
            nv_ptx.WGMMA("Hopper", "m64n8k16.f32.f16.f16"),
        ),
        Case(
            "nv.blackwell.mxfp8",
            32,
            torch.float8_e5m2,
            nv_ptx.TCGen05MMABlockScale(
                "Blackwell", "m128n8k32.block32.f32.e5m2.e5m2.ue8m0"
            ),
            kind="scaled",
            scale_block=32,
        ),
        Case(
            "nv.blackwell.nvfp4",
            64,
            torch.float32,
            nv_ptx.TCGen05MMABlockScale(
                "Blackwell", "m128n8k64.block16.f32.e2m1.e2m1.ue4m3"
            ),
            kind="group_scaled",
            scale_block=16,
        ),
        Case(
            "amd.cdna2.bf16",
            4,
            torch.bfloat16,
            amd.MFMA("CDNA2", "f32_32x32x4bf16"),
        ),
        Case(
            "amd.cdna2.f16",
            4,
            torch.float16,
            amd.MFMA("CDNA2", "f32_32x32x8f16"),
        ),
        Case(
            "amd.cdna3.f16",
            8,
            torch.float16,
            amd.MFMA("CDNA3", "f32_32x32x8_f16"),
        ),
        Case(
            "amd.cdna3.bf16",
            8,
            torch.bfloat16,
            amd.MFMA("CDNA3", "f32_32x32x8_bf16"),
        ),
        Case(
            "amd.cdna3.bf8",
            16,
            torch.float8_e5m2fnuz,
            amd.MFMA("CDNA3", "f32_32x32x16_bf8_bf8"),
        ),
        Case(
            "fp64.fma",
            4,
            torch.float64,
            nv_ptx.MMA("Ampere", "m8n8k4.f64.f64.f64.f64"),
        ),
    ]


def select_cases(cases: Sequence[Case], filters: Sequence[str]) -> list[Case]:
    """Select cases whose names contain any requested substring."""
    if not filters:
        return list(cases)
    selected = [case for case in cases if any(f in case.name for f in filters)]
    if not selected:
        choices = ", ".join(case.name for case in cases)
        raise SystemExit(f"No design matched {list(filters)!r}. Choices: {choices}")
    return selected


def _source_values(batch: int, seed: int, length: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((batch, length), dtype=torch.float64, generator=generator) * 0.5


def _operand_views(case: Case, source: Tensor) -> tuple[Tensor, Tensor]:
    """Return (MMA-Sim storage view, FPy C++ compute view)."""
    storage = source.to(case.storage_dtype)
    if case.tf32:
        return storage, _quantize_tf32(storage)
    if case.storage_dtype == torch.float64:
        return storage, storage
    return storage, storage.float()


def _block_scaled_operand(case: Case, source: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return (MMA-Sim values, FPy values, scales) from TorchAO block quantization."""
    source = source.float().contiguous()
    if case.kind == "scaled":
        scales, values = to_mx(source, case.storage_dtype, case.scale_block)
        return values, values.float(), scales
    # Single-level NVFP4: the per-tensor scale lies outside the MMA.
    scales, packed = nvfp4_quantize(source, case.scale_block)
    values = f4_unpacked_to_f32(unpack_uint4(packed))
    return values, values, scales


def prepare_case(case: Case, batch: int, seed: int, length: int = 1024) -> Prepared:
    """Prepare full operands and chunk views outside the timed region.

    Block-scaled designs carry one scale per block, sliced per model-sized
    chunk for MMA-Sim. Both implementations start each invocation at zero and
    preserve the model's accumulator rounding per chunk.
    """
    if length <= 0 or length % case.k:
        raise ValueError(f"length must be a positive multiple of {case.k}")
    a_source = _source_values(batch, seed, length)
    b_source = _source_values(batch, seed + 1, length)
    if case.kind == "dpa":
        a_mmasim, a_fpy = _operand_views(case, a_source)
        b_mmasim, b_fpy = _operand_views(case, b_source)
        operands = [(a_mmasim, case.k), (b_mmasim, case.k)]
        fpy_scales = ()
    elif case.kind in ("scaled", "group_scaled"):
        a_mmasim, a_fpy, a_scales = _block_scaled_operand(case, a_source)
        b_mmasim, b_fpy, b_scales = _block_scaled_operand(case, b_source)
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

    return Prepared(
        fpy=lambda: fpy_op(a_fpy, b_fpy, *fpy_scales),
        mmasim=mmasim_dot,
    )


def _integer_view(tensor: Tensor) -> Tensor:
    tensor = tensor.detach().contiguous()
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    if tensor.dtype == torch.float64:
        return tensor.view(torch.int64)
    raise TypeError(f"cannot compare bit patterns for {tensor.dtype}")


def validate(prepared: Prepared) -> dict[str, Any]:
    """Return a serializable bitwise-validation result."""
    with torch.no_grad():
        actual = prepared.fpy()
        expected = prepared.mmasim()
    expected_bits = _integer_view(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "valid": False,
            "implementation": "fpy-core",
            "reason": "shape or dtype mismatch",
            "fpy_shape": list(actual.shape),
            "mmasim_shape": list(expected.shape),
            "fpy_dtype": str(actual.dtype),
            "mmasim_dtype": str(expected.dtype),
        }

    actual_bits = _integer_view(actual)
    differences = (actual_bits != expected_bits).nonzero()
    if differences.numel() == 0:
        return {"valid": True}

    flat_index = int(differences[0].item())
    width = 32 if actual.dtype == torch.float32 else 64
    mask = (1 << width) - 1
    return {
        "valid": False,
        "implementation": "fpy-core",
        "reason": "output bit mismatch",
        "first_difference": flat_index,
        "fpy_value": float(actual.flatten()[flat_index]),
        "mmasim_value": float(expected.flatten()[flat_index]),
        "fpy_bits": (
            f"0x{int(actual_bits.flatten()[flat_index]) & mask:0{width // 4}x}"
        ),
        "mmasim_bits": (
            f"0x{int(expected_bits.flatten()[flat_index]) & mask:0{width // 4}x}"
        ),
    }


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
        },
        "parameters": {
            "design_filters": args.design,
            "batch_sizes": args.batch_sizes,
            "dot_lengths": args.dot_lengths,
            "initial_accumulator": 0,
            "inputs": "normal(0, 0.5); MXFP8/NVFP4 block-quantized with TorchAO",
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
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--min-run-time", type=float, default=0.25)
    parser.add_argument("--validate-only", action="store_true")
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
        {**entry, "status": "unsupported"} for entry in UNSUPPORTED
    ]
    rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    started = time.monotonic()

    configurations = product(enumerate(cases), args.dot_lengths, args.batch_sizes)
    for (case_index, case), length, batch in configurations:
        case_seed = args.seed + case_index * 1_000_003 + batch
        prepared = prepare_case(case, batch, case_seed, length)
        result = validate(prepared)
        validation = {
            "design": case.name,
            "instruction": f"{case.reference.arch} {case.reference.shape_and_type}",
            "batch_size": batch,
            "dot_length": length,
            "model_width": case.k,
            "seed": case_seed,
            **result,
        }
        validations.append(validation)
        label = f"{case.name} batch={batch} length={length}"
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

        prepared.fpy()
        prepared.mmasim()
        implementations = [
            ("fpy-core", prepared.fpy),
            ("mmasim", prepared.mmasim),
        ]
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
    write_json(output_dir / "metadata.json", metadata)
    print(f"\nResults: {output_dir}")
    return 1 if metadata["invalid_configurations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
