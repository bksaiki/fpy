# CPU performance

How fast FPy's C++ for each MMA-Sim design (`common/mma/compile.py`'s
`DESIGNS`, as `common/fpy2_models`' operators) runs against MMA-Sim itself on
the CPU.  Each configuration is a batch of dot products, chained one K-wide
call at a time from Python as MMA-Sim's `dpa` is: by
`torch.ops.fpy2_models.<op>` and by MMA-Sim.  Both run on PyTorch's intra-op
threads, as many as `-j` (default: PyTorch's, the performance cores).
Every configuration is first checked to agree with MMA-Sim bit for bit.

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

The designs compile first (`common/fpy2_models/generate_models.py`), into
`build/` under the results, which later runs reuse.  On macOS, MMA-Sim's FMA
helper needs OpenMP, which Apple's clang lacks: it compiles with Homebrew's
LLVM (`brew install llvm`) unless `$CXX` is set.  `--compile` also times
MMA-Sim under default `torch.compile`.

It writes into `evaluation/results/cpu_performance/` (or `-o DIR`):

- `validation.json`: each configuration's agreement with MMA-Sim, and the
  designs FPy's C++ backend refuses;
- `measurements.csv`: each timed block's seconds per call;
- `summary.csv`: each configuration's median, spread and throughput;
- `provenance.json`: the commit, machine, versions and settings;
- `report.py`'s outputs, below.

It exits 1 if any configuration disagrees with MMA-Sim.

## Report

`report.py RESULTS` redraws, at the largest batch size and dot length (or
`--batch`, `--length`):

- `perf-cpu-speedup.{pdf,png}`: each design's speedup over eager MMA-Sim,
  colored by hardware family;
- `perf-cpu-speedup.tex`: each family's geometric mean, least and greatest
  speedup;
- `perf-cpu-paper.json`: every speedup, at every batch size and length;
- `perf-cpu-summary.txt`: the per-design table.

## Tests

```sh
uv run pytest cpu_performance/tests
```

They build the designs once into `.torch_extensions/cpu_performance/`.
