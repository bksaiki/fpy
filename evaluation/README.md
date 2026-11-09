# Evaluation

The paper's evaluation. `run.py` runs all of it; `config.py` holds what each
part measures by default.

| Name | Measures | Needs | Time |
|---|---|---|---|
| `analysis` | how precise format inference is, with and without digit bounds, on the MMA-Sim designs | CPU | ~7 min |
| `sensitivity` | how compile time depends on digit-bound analysis's solver backend (z3, z3's `Optimize`, cvc5) on the MMA-Sim designs | CPU | ~25 min |
| `blas` | each grid design's GEMM backward error on Grading-the-BLAS matrices | CPU, C++ compiler | not yet timed |
| `gsm8k` | GSM8K speed and accuracy: FPy's kernels against MMA-Sim and FP32 | CUDA GPU | ~3.5 h |
| `llm` | how far each grid design moves Qwen3.5-0.8B's next-token predictions | CUDA GPU | ~13 h (TITAN V) |

## Setup

With [uv](https://docs.astral.sh/uv/), from this directory:

```sh
uv sync
```

This installs the locked environment (`pyproject.toml`, `uv.lock`) into
`.venv/`: FPy from this repository, and the exact versions the results were
made with. Nothing needs building beforehand: BLAS's C++ operators and the
Triton kernels compile on first use, cached under the results.

- **Linux x86_64** runs everything; `gsm8k` and `llm` need a CUDA GPU.
- **macOS (Apple silicon)** runs `analysis`, `sensitivity` and `blas`.
- `blas` needs a C++ compiler: `c++`, or `$CXX`.
- torch is its CUDA 12.6 build, the newest that supports Volta (the TITAN V
  the results ran on). Blackwell GPUs need CUDA 12.8 or later: change the
  `pytorch-cu126` index in `pyproject.toml`.
- Do not install `causal-conv1d` or `flash-linear-attention`: `transformers`
  would then run Qwen3.5's linear attention through them, and every `llm`
  result would move.

## Running

```sh
uv run run.py results                                 # everything, ~17 h
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
results/sensitivity/  sensitivity-totals.tex, sensitivity-scatter.pdf, sensitivity-paper.json
results/blas/      blas-grid.csv, plots/k.pdf
results/gsm8k/     gsm8k-speed.tex, gsm8k-speed.json
results/llm/       sweep/sweep-grid-Qwen3.5-0.8B.csv, plots/sweep-grid-Qwen3.5-0.8B.pdf
```

## Tests

```sh
uv run pytest -n 8 analysis/tests common/mma/tests
uv run pytest -n 8 sensitivity/tests                # apart from analysis/tests
uv run pytest gsm8k/tests                           # GPU; apart from analysis/tests
uv run pytest common/mma/core/tests dse/llm/tests   # GPU, serially
```

## Layout

```
run.py, config.py   the entry point, and each evaluation's defaults
analysis/           analysis: format inference's precision
sensitivity/        sensitivity: compile time under each digit-bound solver backend
gsm8k/              gsm8k: speed and accuracy on GSM8K
dse/llm/, dse/blas/ llm, blas: the design space, through an LLM and a GEMM
common/mma/         shared: the designs, their compilers and the harness running them
common/fpy2_models/ shared: the designs as Torch C++ operators
```

Each directory's README or `writeup.md` has the details.
