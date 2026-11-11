# Evaluation

The paper's evaluation. `run.py` runs all of it; `config.py` holds what each
part measures by default.

| Name | Measures | Needs | Time |
|---|---|---|---|
| `analysis` | how precise format inference is, with and without digit bounds, on the MMA-Sim designs | CPU | ~7 min |
| `sensitivity` | how compile time depends on digit-bound analysis's solver backend (z3, z3's `Optimize`, cvc5) on the MMA-Sim designs | CPU | ~25 min |
| `compilation` | how long FPy takes to generate each MMA-Sim design's C++ and Triton code (not the C++ compiler's or Triton's own compilation), and the constraints and queries digit-bound analysis gives its solver | CPU | not yet timed |
| `ablation` | which MMA-Sim designs compile with digit-bound analysis and each set of rewrites switched off | CPU | ~15 min |
| `blas` | each grid design's GEMM backward error on Grading-the-BLAS matrices | CPU, C++ compiler | not yet timed |
| `cpu_performance` | how fast FPy's C++ for each MMA-Sim design runs against MMA-Sim | CPU, C++ compiler | ~1.5 h (M3 Pro) |
| `gsm8k` | GSM8K speed: FPy's kernels against MMA-Sim (tuned: no host waits, CUDA graphs in decode; `--mmasim-as released` for MMA-Sim as released; predicted per layer, `gsm8k/sample.py`) and FP32, every design; MMA-Sim measured whole: `gsm8k/run.py --mmasim full nv.hopper.bf16.f32` (~2.5 h, apart) | CUDA GPU | ~2.5 h (~1.2 h reusing FPy's records: `--reuse DIR`) |
| `llm` | how far each grid design moves Qwen3.5-0.8B's next-token predictions | CUDA GPU | ~13 h (TITAN V) |

## Setup

With [uv](https://docs.astral.sh/uv/), from this directory:

```sh
uv sync
```

This installs the locked environment (`pyproject.toml`, `uv.lock`) into
`.venv/`: FPy from this repository, and the exact versions the results were
made with. Nothing needs building beforehand: BLAS's and CPU performance's
C++ operators and the Triton kernels compile on first use, cached under the
results.

- **Linux x86_64** runs everything; `gsm8k` and `llm` need a CUDA GPU.
- **macOS (Apple silicon)** runs `analysis`, `sensitivity`, `compilation`, `ablation`, `blas` and `cpu_performance`.
- `blas` and `cpu_performance` need a C++ compiler: `c++`, or `$CXX`. On
  macOS, `cpu_performance` also needs Homebrew's LLVM (`brew install llvm`),
  for OpenMP in MMA-Sim's FMA helper.
- torch is its CUDA 12.6 build, the newest that supports Volta (the TITAN V
  the results ran on). Blackwell GPUs need CUDA 12.8 or later: change the
  `pytorch-cu126` index in `pyproject.toml`.
- Do not install `causal-conv1d` or `flash-linear-attention`: `transformers`
  would then run Qwen3.5's linear attention through them, and every `llm`
  result would move.

## Running

```sh
uv run run.py results                                 # everything, ~18.5 h
uv run run.py results --disable gsm8k --disable llm   # the CPU evaluations
uv run run.py results --plot                          # only redraw, from existing results
uv run run.py results --no-cache                      # discard results and caches, rerun
uv run run.py results --threads 8                     # processes or threads per evaluation
```

Each evaluation writes to `results/<name>/`. Before anything runs, `run.py`
checks that the machine has what each evaluation needs. An evaluation whose
results are complete is skipped and only redrawn; after changing `config.py`,
rerun what it affects with `--no-cache`. The model and datasets download from
Hugging Face on first use.

```
results/analysis/  precision-*.tex, precision-slack.pdf, precision-paper.json
results/sensitivity/  sensitivity-table.tex, sensitivity-scatter.pdf, sensitivity-paper.json
results/compilation/  compile-times.csv, compile-times-paper.json, plots/{arch,format,models,solid}.pdf
results/ablation/  ablation-table.tex, ablation-paper.json
results/blas/      blas-grid.csv, plots/k.pdf
results/cpu_performance/  summary.csv, perf-cpu-speedup.pdf, perf-cpu-speedup.tex
results/gsm8k/     gsm8k-speed.tex, gsm8k-designs.tex, gsm8k-check.tex, gsm8k-speed.json, gsm8k-designs.pdf (gsm8k/plot.py)
results/llm/       sweep/sweep-grid-Qwen3.5-0.8B.csv, plots/sweep-grid-Qwen3.5-0.8B.pdf
```

## Tests

```sh
uv run pytest -n 8 analysis/tests common/mma/tests
uv run pytest -n 8 sensitivity/tests                # apart from analysis/tests
uv run pytest cpu_performance/tests                 # builds the designs first
uv run pytest gsm8k/tests                           # GPU; apart from analysis/tests
uv run pytest common/mma/core/tests dse/llm/tests   # GPU, serially
```

## Layout

```
run.py, config.py   the entry point, and each evaluation's defaults
analysis/           analysis: format inference's precision
sensitivity/        sensitivity: compile time under each digit-bound solver backend
compilation/        compilation: FPy's time to generate each design's C++ and Triton
ablation/           ablation: which designs compile with parts of the compiler off
gsm8k/              gsm8k: speed and accuracy on GSM8K
dse/llm/, dse/blas/ llm, blas: the design space, through an LLM and a GEMM
cpu_performance/    cpu_performance: FPy's C++ against MMA-Sim, on the CPU
common/mma/         shared: the designs, their compilers and the harness running them
common/fpy2_models/ shared: the designs as Torch C++ operators
```

Each directory's README or `writeup.md` has the details.
