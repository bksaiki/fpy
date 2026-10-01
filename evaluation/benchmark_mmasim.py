"""Benchmark compiled FPy MMA arithmetic kernels against eager MMA-Sim."""

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
    from fpy2_models.quantization import quantize_e2m1
except ImportError as exc:
    raise SystemExit(
        "Could not import the fpy2_models extension. Build it first with:\n"
        "  cd cpp_extension && ../.venv/bin/python setup.py build_ext --inplace"
    ) from exc


@dataclass(frozen=True)
class Case:
    """One pair of equivalent FPy and MMA-Sim arithmetic kernels."""

    name: str
    op_name: str
    k: int
    storage_dtype: torch.dtype
    accumulator_dtype: torch.dtype
    reference: object
    kind: str = "dpa"
    tf32: bool = False
    scale_dtype: torch.dtype | None = None
    scale_count: int = 0
    operand_format: str = ""


@dataclass(frozen=True)
class Prepared:
    """Inputs and zero-argument callables for one timed configuration."""

    fpy: Callable[[], Tensor]
    fpy_storage: Callable[[], Tensor]
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
        from mmasim.arithmetic import fdpa, fma, ftz_mul_add
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "Could not import the local MMA-Sim arithmetic models. On macOS, "
            "MMA-Sim's FP64 helper requires an OpenMP-capable compiler; install "
            "Homebrew LLVM or set CXX to one before running the benchmark.\n"
            f"Original error: {exc}"
        ) from exc

    return [
        Case(
            "nv.volta.f16.f32",
            "nv_volta_f16_f32",
            8,
            torch.float16,
            torch.float32,
            fdpa.MMA_T_FDPA(23, "RZ-FP32", 4, -131),
            operand_format="fp16",
        ),
        Case(
            "nv.turing.f16.f32",
            "nv_turing_f16_f32",
            16,
            torch.float16,
            torch.float32,
            fdpa.MMA_T_FDPA(24, "RZ-FP32", 8, -132),
            operand_format="fp16",
        ),
        Case(
            "nv.ampere.tf32.f32",
            "nv_ampere_tf32_f32",
            8,
            torch.float32,
            torch.float32,
            fdpa.MMA_T_FDPA(24, "RZ-FP32", 4, -132),
            tf32=True,
            operand_format="tf32",
        ),
        Case(
            "nv.ampere.bf16.f32",
            "nv_ampere_bf16_f32",
            16,
            torch.bfloat16,
            torch.float32,
            fdpa.MMA_T_FDPA(24, "RZ-FP32", 8, -132),
            operand_format="bf16",
        ),
        Case(
            "nv.ada.e5m2.f32",
            "nv_ada_e5m2_f32",
            16,
            torch.float8_e5m2,
            torch.float32,
            fdpa.MMA_T_FDPA(13, "RZ-E8M13", 16, -132),
            operand_format="e5m2",
        ),
        Case(
            "nv.hopper.f16.f32",
            "nv_hopper_f16_f32",
            32,
            torch.float16,
            torch.float32,
            fdpa.MMA_T_FDPA(25, "RZ-FP32", 16, -133),
            operand_format="fp16",
        ),
        Case(
            "nv.blackwell.mxfp8",
            "nv_blackwell_mxfp8",
            32,
            torch.float8_e5m2,
            torch.float32,
            fdpa.MMA_ST_FDPA(25, "RZ-FP32", 32, -133),
            kind="scaled",
            scale_dtype=torch.float8_e8m0fnu,
            scale_count=1,
            operand_format="e5m2",
        ),
        Case(
            "nv.blackwell.nvfp4",
            "nv_blackwell_nvfp4",
            64,
            torch.float32,
            torch.float32,
            fdpa.MMA_GST_FDPA(16, 35, "RZ-FP32", 64, -139),
            kind="group_scaled",
            scale_dtype=torch.float8_e4m3fn,
            scale_count=4,
            operand_format="e2m1",
        ),
        Case(
            "amd.cdna2.bf16",
            "amd_cdna2_bf16",
            4,
            torch.bfloat16,
            torch.float32,
            ftz_mul_add.MMA_FTZ_MUL_ADD(2),
            operand_format="bf16",
        ),
        Case(
            "amd.cdna2.f16",
            "amd_cdna2_f16",
            4,
            torch.float16,
            torch.float32,
            ftz_mul_add.MMA_FTZ_MUL_ADD(4),
            operand_format="fp16",
        ),
        Case(
            "amd.cdna3.f16",
            "amd_cdna3_f16",
            8,
            torch.float16,
            torch.float32,
            fdpa.MMA_TR_FDPA(24, 31, "RNE-FP32", 8),
            operand_format="fp16",
        ),
        Case(
            "amd.cdna3.bf16",
            "amd_cdna3_bf16",
            8,
            torch.bfloat16,
            torch.float32,
            fdpa.MMA_TR_FDPA(24, 31, "RNE-FP32", 8),
            operand_format="bf16",
        ),
        Case(
            "amd.cdna3.bf8",
            "amd_cdna3_bf8",
            16,
            torch.float8_e5m2fnuz,
            torch.float32,
            fdpa.MMA_GTR_FDPA(24, 31, "RNE-FP32", 16),
            operand_format="bf8",
        ),
        Case(
            "fp64.fma",
            "fp64_fma",
            4,
            torch.float64,
            torch.float64,
            fma.MMA_FMA(),
            operand_format="fp64",
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


def _source_values(case: Case, batch: int, workload: str, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    if workload == "normal":
        return (
            torch.randn((batch, case.k), dtype=torch.float64, generator=generator) * 0.5
        )

    if case.operand_format == "e2m1":
        tiny = 0.5
    else:
        tiny = float(torch.finfo(case.storage_dtype).smallest_normal)
    values = torch.tensor(
        [0.0, -0.0, tiny / 2.0, -tiny / 2.0, tiny, -tiny, 0.5, -0.5],
        dtype=torch.float64,
    )
    offset = int(torch.randint(len(values), (), generator=generator).item())
    indices = (torch.arange(batch * case.k) + offset) % len(values)
    return values[indices].reshape(batch, case.k)


def _operand_views(case: Case, source: Tensor) -> tuple[Tensor, Tensor]:
    """Return (MMA-Sim storage view, FPy C++ compute view)."""
    if case.operand_format == "e2m1":
        quantized = quantize_e2m1(source)
        return quantized, quantized

    storage = source.to(case.storage_dtype)
    if case.tf32:
        return storage, _quantize_tf32(storage)
    if case.storage_dtype == torch.float64:
        return storage, storage
    return storage, storage.float()


def prepare_case(
    case: Case,
    batch: int,
    workload: str,
    seed: int,
) -> Prepared:
    """Prepare equivalent inputs and closures without timing setup work."""
    a_source = _source_values(case, batch, workload, seed)
    b_source = _source_values(case, batch, workload, seed + 1)
    a_mmasim, a_fpy = _operand_views(case, a_source)
    b_mmasim, b_fpy = _operand_views(case, b_source)

    c_value = 0.125
    c_mmasim = torch.full((batch,), c_value, dtype=case.accumulator_dtype)
    c_fpy = torch.tensor(c_value, dtype=case.accumulator_dtype).item()
    fpy_op = getattr(torch.ops.fpy2_models, case.op_name).default
    fpy_storage_op = getattr(fpy2_models, case.op_name)

    if case.kind == "dpa":
        return Prepared(
            fpy=lambda: fpy_op(a_fpy, b_fpy, c_fpy),
            fpy_storage=lambda: fpy_storage_op(a_mmasim, b_mmasim, c_fpy),
            mmasim=lambda: case.reference.dpa(a_mmasim, b_mmasim, c_mmasim),
        )

    if case.kind == "scaled":
        assert case.scale_dtype is not None
        alpha_fpy, beta_fpy = 1.0, 2.0
        alpha_mmasim = torch.full((batch, 1), alpha_fpy, dtype=case.scale_dtype)
        beta_mmasim = torch.full((batch, 1), beta_fpy, dtype=case.scale_dtype)
        return Prepared(
            fpy=lambda: fpy_op(a_fpy, b_fpy, c_fpy, alpha_fpy, beta_fpy),
            fpy_storage=lambda: fpy_storage_op(
                a_mmasim,
                b_mmasim,
                c_fpy,
                alpha_fpy,
                beta_fpy,
            ),
            mmasim=lambda: case.reference.dpa(
                a_mmasim, b_mmasim, c_mmasim, alpha_mmasim, beta_mmasim
            ),
        )

    if case.kind == "group_scaled":
        assert case.scale_dtype is not None
        scale_source = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
        alpha_storage = scale_source.to(case.scale_dtype).repeat(batch, 1)
        beta_storage = scale_source.flip(0).to(case.scale_dtype).repeat(batch, 1)
        alpha_fpy = alpha_storage.float()
        beta_fpy = beta_storage.float()
        return Prepared(
            fpy=lambda: fpy_op(a_fpy, b_fpy, c_fpy, alpha_fpy, beta_fpy),
            fpy_storage=lambda: fpy_storage_op(
                a_mmasim,
                b_mmasim,
                c_fpy,
                alpha_storage,
                beta_storage,
            ),
            mmasim=lambda: case.reference.dpa(
                a_mmasim,
                b_mmasim,
                c_mmasim,
                alpha_storage,
                beta_storage,
            ),
        )

    raise AssertionError(f"unknown case kind: {case.kind}")


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
        storage_actual = prepared.fpy_storage()
        expected = prepared.mmasim()
    expected_bits = _integer_view(expected)
    for implementation, candidate in (
        ("fpy-core", actual),
        ("fpy-storage", storage_actual),
    ):
        if candidate.shape != expected.shape or candidate.dtype != expected.dtype:
            return {
                "valid": False,
                "implementation": implementation,
                "reason": "shape or dtype mismatch",
                "fpy_shape": list(candidate.shape),
                "mmasim_shape": list(expected.shape),
                "fpy_dtype": str(candidate.dtype),
                "mmasim_dtype": str(expected.dtype),
            }

        candidate_bits = _integer_view(candidate)
        differences = (candidate_bits != expected_bits).nonzero()
        if differences.numel() == 0:
            continue

        flat_index = int(differences[0].item())
        width = 32 if candidate.dtype == torch.float32 else 64
        mask = (1 << width) - 1
        return {
            "valid": False,
            "implementation": implementation,
            "reason": "output bit mismatch",
            "first_difference": flat_index,
            "fpy_value": float(candidate.flatten()[flat_index]),
            "mmasim_value": float(expected.flatten()[flat_index]),
            "fpy_bits": (
                f"0x{int(candidate_bits.flatten()[flat_index]) & mask:0{width // 4}x}"
            ),
            "mmasim_bits": (
                f"0x{int(expected_bits.flatten()[flat_index]) & mask:0{width // 4}x}"
            ),
        }
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
    workload: str,
    batch: int,
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
                "workload": workload,
                "batch_size": batch,
                "threads": threads,
                "replicate": replicate,
                "sample": sample,
                "number_per_run": measurement.number_per_run,
                "raw_seconds": raw_seconds,
                "seconds_per_call": seconds_per_call,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["design"],
            row["implementation"],
            row["workload"],
            row["batch_size"],
            row["threads"],
        )
        groups[key].append(row["seconds_per_call"])

    summaries: list[dict[str, Any]] = []
    for key, values in groups.items():
        design, implementation, workload, batch, threads = key
        median, q1, iqr = _quartiles(values)
        summaries.append(
            {
                "design": design,
                "implementation": implementation,
                "workload": workload,
                "batch_size": batch,
                "threads": threads,
                "samples": len(values),
                "median_seconds_per_call": median,
                "q1_seconds_per_call": q1,
                "iqr_seconds_per_call": iqr,
                "relative_iqr": iqr / median,
                "ns_per_dpa": median * 1e9 / batch,
                "dpa_per_second": batch / median,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["workload"],
            row["design"],
            row["batch_size"],
            row["implementation"],
        ),
    )


def enrich_measurements(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> None:
    """Attach configuration aggregates to every normalized raw sample."""
    by_key = {
        (
            row["design"],
            row["implementation"],
            row["workload"],
            row["batch_size"],
            row["threads"],
        ): row
        for row in summaries
    }
    for row in rows:
        summary = by_key[
            (
                row["design"],
                row["implementation"],
                row["workload"],
                row["batch_size"],
                row["threads"],
            )
        ]
        for field in (
            "median_seconds_per_call",
            "q1_seconds_per_call",
            "iqr_seconds_per_call",
            "relative_iqr",
            "ns_per_dpa",
            "dpa_per_second",
        ):
            row[field] = summary[field]


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
            "workloads": args.workloads,
            "seed": args.seed,
            "replicates": args.replicates,
            "min_run_time": args.min_run_time,
        },
    }


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


def parse_workloads(value: str) -> list[str]:
    values = value.split(",")
    unknown = set(values) - {"normal", "stress"}
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown workloads: {sorted(unknown)}")
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
        default=parse_int_list("1,16,256,4096,65536"),
    )
    parser.add_argument(
        "--workloads", type=parse_workloads, default=parse_workloads("normal,stress")
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

    if args.output is None:
        run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output_dir = EVALUATION_ROOT / "results" / run_id
    else:
        output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = collect_metadata(args, mmasim_root)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    validations: list[dict[str, Any]] = [
        {**entry, "status": "unsupported"} for entry in UNSUPPORTED
    ]
    rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    started = time.monotonic()

    for case_index, case in enumerate(cases):
        for workload in args.workloads:
            for batch in args.batch_sizes:
                case_seed = args.seed + case_index * 1_000_003 + batch
                prepared = prepare_case(case, batch, workload, case_seed)
                result = validate(prepared)
                validation = {
                    "design": case.name,
                    "workload": workload,
                    "batch_size": batch,
                    "seed": case_seed,
                    **result,
                }
                validations.append(validation)
                if not result["valid"]:
                    print(
                        f"INVALID {case.name} workload={workload} batch={batch}: "
                        f"{result['reason']}",
                        flush=True,
                    )
                    continue
                if args.validate_only:
                    print(
                        f"VALID   {case.name} workload={workload} batch={batch}",
                        flush=True,
                    )
                    continue

                prepared.fpy()
                prepared.fpy_storage()
                prepared.mmasim()
                implementations = [
                    ("fpy-core", prepared.fpy),
                    ("fpy-storage", prepared.fpy_storage),
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
                                workload=workload,
                                batch=batch,
                                threads=args.threads,
                                replicate=replicate,
                                min_run_time=args.min_run_time,
                                rng=rng,
                            )
                        )
                print(
                    f"TIMED   {case.name} workload={workload} batch={batch}",
                    flush=True,
                )

    (output_dir / "validation.json").write_text(
        json.dumps(validations, indent=2, sort_keys=True) + "\n"
    )
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
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nResults: {output_dir}")
    return 1 if metadata["invalid_configurations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
