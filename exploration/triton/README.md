# Triton spike — the flag audit behind `docs/todos/backend-triton.md`

Hand-written Triton kernels for a known FPy program, run against the
interpreter, to pin the numerics flags *before* an emitter exists.  The point
of doing it first is that it can invalidate the rest of the roadmap for about a
week's cost, and that it leaves the target text known before anything has to
generate it.

```sh
python -m exploration.triton.check            # everything runnable here
python -m exploration.triton.check --n 4000   # more samples
```

| File | Contents |
|---|---|
| `programs.py` | The FPy programs: a dot product whose products are exact, one whose are not, and an `fp.fma` twin of each |
| `kernels.py` | What the backend would have to emit for them — batch-lifted, one lane per dot product |
| `check.py` | The differential harness |

## Findings

Every prediction below held on an **NVIDIA TITAN V (sm_70)**, torch 2.14.0+cu126,
triton 3.8.0, at `--n 2000`.

### 1. `enable_fp_fusion` is derivable, not a global pin

An early draft of the roadmap said to pin `enable_fp_fusion=False`. That is
too blunt.

Contracting `acc + x * y` into an `fma` computes the product exactly and rounds
once; the unfused form rounds the product and then rounds the sum. Where the
product is *already* exact those are the same operation. Predicted from FPy's
semantics alone — each program beside its `fp.fma` twin, no GPU — then
confirmed against the real flag:

| Program | interpreter: fma twin | hardware: `enable_fp_fusion=True` | `=False` |
|---|---|---|---|
| FP16 in, exact products, FP32 accumulate | identical **0/2000** | **0/2000** differ | 0/2000 |
| FP32 throughout, product rounds | differs **609/2000** | **590/2000** differ | 0/2000 |

So fusion is safe exactly where the product fits the storage chosen for it —
`scalar_fits_in(product_format, product_storage)`, which the pipeline already
computes for every expression. The backend can emit `enable_fp_fusion=True` per
kernel where the analysis discharges it, rather than paying the reported ~30%
cost of `--fmad=false` everywhere.

A decision the target description makes from an existing analysis, not a flag a
user sets — and the first concrete case of this backend knowing something
Triton never asks about.

### 2. The fp16 cast trap is real, and total

Triton types `fp16 op fp16` as fp16, so `x * y` on two fp16 operands rounds the
product the program requires to be exact. `kernels.dot_trap` is that wrong
lowering, kept executable:

| Kernel | vs interpreter |
|---|---|
| `dot_exact` — `x.to(tl.float32) * y.to(tl.float32)` | **0/2000** differ |
| `dot_trap` — `(x * y).to(tl.float32)` | **2000/2000** differ |

Total, not marginal. The casts come from `StorageInfer` giving the product fp32
storage — exactly as the C++ backend already emits `static_cast<double>` — and
nothing in Triton asks for them.

### 3. A hand-written batch-lifted kernel is bit-exact

`dot_exact` and `dot_fp32` both match the interpreter bit-for-bit on every
sample. One lane per dot product, the fold sequential *within* a lane: full
parallelism with FPy's left-fold order preserved, which is the roadmap's
argument for sequencing §7 ahead of §8.

### 4. Out of scope, so not tested

`bf16` and TF32 were dropped from the target. This box could not have tested
them anyway — Volta has neither.

Still open, and untestable here: whether `enable_fp_fusion=False` reaches the
packed `mul.rn.f32x2` / `add.rn.f32x2` the backend emits on **Blackwell**
(sm_100). Given finding 1 the question is narrower than it was — it only bites
where the flag is needed, i.e. where the product rounds — but it would be a
silent bit-exactness hole rather than a refusal, so it wants checking on such a
card before anyone trusts the emitter there.

## Environment

Two requirements beyond `uv sync`, both of which fail in confusing ways.

**A `uv`-managed interpreter.** `triton` compiles a small driver shim with
`gcc` at first kernel launch, so it needs Python development headers. A venv
built on a distro interpreter that ships none — `/usr/include/pythonX.Y`
missing — dies with a `CalledProcessError` from `gcc`, or misleadingly with
`ValueError: @jit functions should be defined in a Python file`. A `uv`-managed
build bundles its headers:

```sh
uv python install 3.14
rm -rf .venv
uv venv --python ~/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14
uv sync
```

`uv venv --python 3.14` alone is not enough if a system 3.14 also satisfies the
request — name the managed interpreter by path. There is no `.python-version`
in this repo, so nothing makes the choice sticky.

**A torch build matching the GPU.** The default `torch==2.14.0` (cu130) starts
at sm_75 and cannot allocate on Volta at all. The `cu126` build of the *same
version* covers sm_50–sm_90:

```sh
uv pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.14.0'
uv pip install triton
```

Neither is in `pyproject.toml` on purpose — this is an `exploration/` spike, and
a GPU dependency does not belong in the project manifest. `uv sync` will remove
them; rerun the two commands above.

Last: `@triton.jit` functions must live in a real file, since Triton reads their
source. That is why `kernels.py` exists rather than the kernels being inline in
the harness.
