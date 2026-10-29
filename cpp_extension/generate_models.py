"""Generate the FPy model translation units and their Torch operators.

Compiles every MMA-Sim design in `examples/mmasim/compile.py` that the C++
backend supports, writes one translation unit per design, and emits:

- `torch_ops_generated.cpp`: for each design, a single-DPA operator `<op>` and
  a complete dot-product operator `<op>_dot`, registered under `fpy2_models`;
- `manifest.json`: each design's name, operator, width, kind, and scales.

Each design's operator signature follows its generated entry function:
`(A, B, c)` is a plain DPA, `(A, B, c, alpha, beta)` with scalar scales an MX
DPA, and with array scales a group-scaled DPA.

    python cpp_extension/generate_models.py OUT_DIR
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT.parent / "examples" / "mmasim"
sys.path.insert(0, str(EXAMPLES))

import compile as designs  # noqa: E402

UNSUPPORTED = {
    # The C++ backend cannot select storage for `e_fdpa_block`.
    "amd.cdna1.bf16": "The FPy C++ backend cannot select storage for e_fdpa_block.",
    "amd.cdna1.f16": "The FPy C++ backend cannot select storage for e_fdpa_block.",
}

_ARRAY = re.compile(r"const std::array<(float|double), (\d+)>& \w+")
_SCALAR = re.compile(r"(float|double) \w+")
_SCALAR_TYPES = {"float": "Float", "double": "Double"}


def _signature(src: str, entry: str) -> tuple[str, list[tuple[str, int | None]]]:
    """The return type and `(type, length or None)` parameters of *entry*."""
    match = re.search(rf"^(float|double) {re.escape(entry)}\((.*)\) {{$", src, re.M)
    if match is None:
        raise ValueError(f"no definition of {entry} in the generated source")
    params = []
    for part in _split_params(match.group(2)):
        if array := _ARRAY.fullmatch(part):
            params.append((array.group(1), int(array.group(2))))
        elif scalar := _SCALAR.fullmatch(part):
            params.append((scalar.group(1), None))
        else:
            raise ValueError(f"unexpected parameter {part!r} of {entry}")
    return match.group(1), params


def _split_params(params: str) -> list[str]:
    """Split a parameter list at commas outside template brackets."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(params):
        depth += (ch == "<") - (ch == ">")
        if ch == "," and depth == 0:
            parts.append(params[start:i].strip())
            start = i + 1
    parts.append(params[start:].strip())
    return parts


def _describe(name: str, src: str, entry: str) -> dict:
    """The manifest entry for one compiled design."""
    result, params = _signature(src, entry)
    (a_type, k), (b_type, k_b), (c_type, c_len), *scales = params
    types = {result, a_type, b_type, c_type, *(t for t, _ in scales)}
    if len(types) != 1 or k != k_b or c_len is not None:
        raise ValueError(f"{name}: unsupported signature {params}")
    if not scales:
        kind, scale_k = "dpa", None
    elif len(scales) == 2 and scales[0] == scales[1]:
        scale_k = scales[0][1]
        kind = "scaled" if scale_k is None else "group_scaled"
    else:
        raise ValueError(f"{name}: unsupported scales {scales}")
    namespace = designs._namespace(name)
    return {
        "name": name,
        "op": namespace.removeprefix("model_"),
        "namespace": namespace,
        "entry": entry,
        "file": designs._filename(name),
        "compute": result,
        "k": k,
        "kind": kind,
        "scale_k": scale_k,
    }


def _declaration(model: dict) -> str:
    t, k = model["compute"], model["k"]
    params = [f"const std::array<{t}, {k}>&"] * 2 + [t]
    if model["kind"] == "scaled":
        params += [t, t]
    elif model["kind"] == "group_scaled":
        params += [f"const std::array<{t}, {model['scale_k']}>&"] * 2
    return (
        f"namespace fpy_models::{model['namespace']} {{\n"
        f"{t} {model['entry']}({', '.join(params)});\n"
        "}\n"
    )


def _wrappers(model: dict) -> str:
    t, k, op = model["compute"], model["k"], model["op"]
    fn = f"&fpy_models::{model['namespace']}::{model['entry']}"
    dtype = f"ScalarType::{_SCALAR_TYPES[t]}"
    if model["kind"] == "dpa":
        return (
            f"Tensor {op}_cpu(const Tensor& a, const Tensor& b, double c) {{\n"
            f"  return run_dpa<{t}, {k}, {fn}>(a, b, c, {dtype});\n}}\n\n"
            f"Tensor {op}_dot_cpu(const Tensor& a, const Tensor& b) {{\n"
            f"  return run_dpa<{t}, {k}, {fn}, true>(a, b, 0.0, {dtype});\n}}\n"
        )
    scale_k = 1 if model["kind"] == "scaled" else model["scale_k"]
    single = (
        f"Tensor {op}_cpu(const Tensor& a, const Tensor& b, double c, "
        "double alpha, double beta) {\n"
        f"  return run_scaled_dpa<{t}, {k}, {fn}>(a, b, c, alpha, beta, {dtype});\n}}\n"
        if model["kind"] == "scaled"
        else f"Tensor {op}_cpu(const Tensor& a, const Tensor& b, double c, "
        "const Tensor& alphas, const Tensor& betas) {\n"
        f"  return run_group_scaled_dpa<{t}, {k}, {scale_k}, {fn}>(\n"
        f"      a, b, c, alphas, betas, {dtype});\n}}\n"
    )
    return single + (
        f"\nTensor {op}_dot_cpu(const Tensor& a, const Tensor& b, "
        "const Tensor& alphas, const Tensor& betas) {\n"
        f"  return run_group_scaled_dpa<{t}, {k}, {scale_k}, {fn}, true>(\n"
        f"      a, b, 0.0, alphas, betas, {dtype});\n}}\n"
    )


def _schemas(model: dict) -> list[str]:
    op = model["op"]
    scales = {
        "dpa": ("", ""),
        "scaled": (", float alpha, float beta", ", Tensor alphas, Tensor betas"),
        "group_scaled": (", Tensor alphas, Tensor betas",) * 2,
    }[model["kind"]]
    return [
        f"{op}(Tensor a, Tensor b, float c{scales[0]}) -> Tensor",
        f"{op}_dot(Tensor a, Tensor b{scales[1]}) -> Tensor",
    ]


def _operators(models: list[dict]) -> str:
    defs = "\n".join(f'  m.def("{s}");' for m in models for s in _schemas(m))
    impls = "\n".join(
        f'  m.impl("{name}", TORCH_BOX(&{name}_cpu));'
        for m in models
        for name in (m["op"], m["op"] + "_dot")
    )
    return (
        "// Generated by cpp_extension/generate_models.py; do not edit.\n"
        '#include "runners.h"\n\n'
        + "\n".join(_declaration(m) for m in models)
        + "\nnamespace {\n\n"
        "using namespace fpy_models::runners;\n"
        "using torch::headeronly::ScalarType;\n"
        "using torch::stable::Tensor;\n\n"
        + "\n".join(_wrappers(m) for m in models)
        + "\n}  // namespace\n\n"
        f"STABLE_TORCH_LIBRARY(fpy2_models, m) {{\n{defs}\n}}\n\n"
        f"STABLE_TORCH_LIBRARY_IMPL(fpy2_models, CPU, m) {{\n{impls}\n}}\n"
    )


def generate(out: Path, jobs: int) -> list[dict]:
    names = [name for name, _ in designs.DESIGNS if name not in UNSUPPORTED]
    builds = dict(designs.DESIGNS)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.cpp"):
        stale.unlink()
    models = []
    results = designs.in_processes(designs._compile_named, names, jobs)
    for name, (src, kind, why, _) in zip(names, results):
        if src is None:
            raise SystemExit(f"{name} does not compile: {kind}: {why}")
        entry = builds[name]()[0].name
        model = _describe(name, src, entry)
        (out / model["file"]).write_text(designs._translation_unit(name, src))
        models.append(model)
    (out / "torch_ops_generated.cpp").write_text(_operators(models))
    manifest = {"models": models, "unsupported": UNSUPPORTED}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return models


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    models = generate(Path(sys.argv[1]), os.cpu_count() or 1)
    print(f"generated {len(models)} models")
