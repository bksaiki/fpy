# CPU performance

How fast FPy's C++ for each MMA-Sim design (`common/mma/compile.py`'s
`DESIGNS`, as `common/fpy2_models`' operators) runs against MMA-Sim itself on
the CPU.  Each configuration is a batch of dot products, chained one K-wide
call at a time from Python as MMA-Sim's `dpa` is: by
`torch.ops.fpy2_models.<op>` and by MMA-Sim tuned, `common/mma/no_sync.py`'s
arithmetic under default `torch.compile`.  Both run on PyTorch's intra-op
threads, as many as `-j` (default: PyTorch's, the performance cores).
Every configuration, both FPy and tuned MMA-Sim, is first checked to agree
with MMA-Sim as released bit for bit.

Operands are drawn from normal(0, 0.5) and rounded to each design's formats;
MX and NVFP4 operands are block-quantized by TorchAO.  Batch sizes, dot
lengths, replicates and the rest are `config.py`'s `CPU_PERF_*`.

## Run

From `evaluation/`, about 1.5 hours on an M3 Pro:

```sh
uv run cpu_performance/run.py
uv run cpu_performance/run.py --smoke                   # one batch and length, briefly
uv run cpu_performance/run.py --designs hopper -j 1     # designs by substring, one thread
```

The designs compile first, as the `common/fpy2_models` package (its
`setup.py`, about a minute), rebuilt every run so that no stale build is
timed.  On macOS, MMA-Sim's FMA
helper needs OpenMP, which Apple's clang lacks: it compiles with Homebrew's
LLVM (`brew install llvm`) unless `$CXX` is set.

It writes into `evaluation/results/cpu_performance/` (or `-o DIR`):

- `validation.json`: each configuration's agreement with MMA-Sim, and the
  designs FPy's C++ backend refuses;
- `measurements.csv`: each timed block's seconds per call;
- `summary.csv`: each configuration's median, spread and throughput;
- `provenance.json`: the commit, machine, versions and settings;
- `report.py`'s outputs, below.

A partial run (`--smoke`, `--designs`, or other `--batches` or `--lengths`)
writes into `evaluation/results/cpu_performance-partial/` instead, so as not
to replace a full run's results.  `summary.csv` is written last, and only if
every configuration agrees with MMA-Sim; otherwise the run exits 1 without it,
and `evaluation/run.py` reruns it next time.

## Report

`report.py RESULTS` redraws, at the largest batch size and dot length (or
`--batch`, `--length`):

- `perf-cpu-speedup.{pdf,png}`: each design's speedup over tuned MMA-Sim,
  colored by hardware family;
- `perf-cpu-speedup.tex`: each family's geometric mean, least and greatest
  speedup;
- `perf-cpu-macros.tex`: the paper's `\EvalCPUPerf...` macros (speedups,
  designs, batch, length, threads), to paste into its `eval.tex`;

and prints the per-design table.

## Tests

```sh
uv run pytest cpu_performance/tests
```

They rebuild the `fpy2_models` package once per session.
