# Fixed-length dot-product evaluation

Compare compiled FPy models against the equivalent eager and `torch.compile`d
MMA-Sim models by
computing complete dot products of lengths **256, 1024, and 4096**, with batches
of **1024, 4096, and 16384** independent dot products by default.

For each design, split each vector into consecutive chunks of the model's
existing width (4–64 terms). Start an accumulator at zero, call the unchanged
DPA model on each chunk with the previous result as its accumulator, and return
the final result. Each model retains its own arithmetic and rounding semantics.
Different designs need not produce the same answer; each FPy design must match
its corresponding eager MMA-Sim design bit for bit before it is timed.

FPy's compiled models are Torch operators with the same interface as
MMA-Sim's `dpa`: operands of shape (batch, K) and one accumulator per row. The
same Python loop calls either one once per chunk. A batch of 16 means 16
independent length-L dot products with separate accumulators. Larger batches
amortize MMA-Sim's dispatch overhead. The comparison measures CPU performance,
not GPU performance. FPy's operator applies its model to one row at a time;
the default is one Torch thread. Optional thread counts affect
Torch/MMA-Sim (including Inductor's generated kernels) rather than
parallelizing the FPy wrapper.

There are three series:

- `fpy-core`: compute-format inputs passed to FPy's compiled model, called
  once per chunk from the same Python loop as MMA-Sim.
- `mmasim`: equivalent storage-format inputs passed to eager MMA-Sim.
- `mmasim-compiled`: the same inputs, with unmodified MMA-Sim's `dpa`
  compiled by `torch.compile` with default settings (Inductor CPU, graph breaks
  allowed) and called in the eager chunk loop. Enable it with `--compile`.

Default `torch.compile` can fail to compile a design or compile it incorrectly.
Each failure is recorded in `validation.json` and excludes only this series
from timing for that configuration; `metadata.json` counts them.

Compilation happens on the first (validation) call and is excluded from
timing; recompilation for each length and batch size takes a few seconds.

Input generation, quantization, chunk preparation, compilation, and correctness
validation are outside timing, as is `torch.compile` compilation. Both paths use the same quantized numerical
operand values, in the physical representations their interfaces require.
MMA-Sim chunk tensors are made contiguous ahead of timing. Timings include
execution, dispatch, accumulator initialization, and output allocation. Every
invocation starts fresh at zero. The former `fpy-quantize` series is removed.

Inputs are sampled in FP64 and rounded to each design's input format. By
default they come from N(0, 0.5²); `--distribution uniform` samples U[-1, 1)
instead. `--dot-lengths` sets the vector length. Block-scaled operands are
quantized with TorchAO: MX designs get one E8M0 scale per 32 elements
(`to_mx`), MXFP4 one E8M0 scale per 16, and NVFP4 one E4M3 scale per 16
(single-level `nvfp4_quantize`; the per-tensor scale lies outside the MMA).
Every chunk receives its own block scales.

The benchmark covers every design in `examples/mmasim/compile.py` that the C++
backend compiles: 60 of 62. The two CDNA1 designs are unsupported, because the
backend cannot select storage for `e_fdpa_block`. Each MMA-Sim model comes from
MMA-Sim's instruction simulators (for example
`nv_ptx.MMA("Volta", "m8n8k4.f32.f16.f16.f32")`), which select the arithmetic
model and its parameters; `_references` in `benchmark_mmasim.py` maps each FPy
design to one, and `validation.json` records the instruction for each design.

MMA-Sim rejects packed E2M1 (FP4) operands everywhere but its group-scaled
model (NVFP4, MXFP4), which unpacks them to FP16 values. The benchmark passes
E2M1 operands to every MMA-Sim model as those FP16 values; all 14 designs with
E2M1 operands match FPy bit for bit. FP16-output designs compute in FP32
storage in C++; their results must be exact FP16 values, and are compared in
FP16.

## Build and run

Build from the repository root:

```sh
(cd evaluation/common/fpy2_models && ../../../.venv/bin/python setup.py build_ext --inplace)
./.venv/bin/python -m pytest evaluation -q
```

The first command builds FPy's models as Torch operators; see
[fpy2_models](../common/fpy2_models/README.md) for what it does. The benchmark
reads the built models' `manifest.json`. Benchmarking and plotting do not
rebuild the FPy models.

The benchmark uses `~/repos/mma-sim` by default; override with
`--mmasim-root PATH`. On macOS it selects Homebrew LLVM when available for
MMA-Sim's OpenMP helper unless `CXX` is already set.

Run the full evaluation:

```sh
./.venv/bin/python evaluation/performance/benchmark_mmasim.py
```

By default it runs a 3 × 3 grid: dot lengths 256, 1024 and 4096 by batch
sizes 1024, 4096 and 16384, on one thread, with three replicates of at least
0.25 seconds each, and no `torch.compile` series. Override them with `--dot-lengths`, `--batch-sizes`,
`--threads`, `--replicates` and `--min-run-time`; add `--compile` for the
`torch.compile` series.

For a short smoke run, add `--design amd.cdna2.f16 --replicates 1
--min-run-time 0.05`. Use `--validate-only` for correctness
without timings. Custom lengths must be positive multiples of every selected
model's width; partial chunks are rejected.

Each run writes a new directory under `evaluation/results/`:

- `metadata.json`: command-line parameters, revisions, versions, hardware and
  compiler.
- `validation.json`: bitwise correctness results for each configuration.
- `measurements.csv`: one row per timed block: the seconds per call in that
  block, with its configuration and replicate.
- `summary.csv`: per configuration, the median and IQR of seconds per call
  over all blocks, nanoseconds per dot product, and dot products per second.

One timed call computes the entire batch; per-dot metrics divide by batch size.
No speedup is reported for a configuration that fails validation.

Generate throughput and speedup plots for each length, with one panel per batch
size. Plots show FPy, eager MMA-Sim, and default `torch.compile` MMA-Sim;
speedups use a log scale:

```sh
./.venv/bin/python evaluation/performance/plot_mmasim.py evaluation/results/<run-id>
```

`--batch-size` and `--dot-length` select headline tables. Plots still cover
all measured lengths. A `!` marks relative IQR above 10%. The plotter expects the
fixed-length result schema; for older runs that also measured a stress workload,
it plots only the normal rows.

## Historical results

[RESULTS.md](RESULTS.md) and [TODO.md](TODO.md) describe the previous independent
single-DPA evaluation. Those measurements do not describe complete fixed-length
dot products and must not be compared directly to the new results.

## Iterating on the evaluation

Data collection and plotting are independent commands. In
`benchmark_mmasim.py`, `mmasim_references` pairs designs with MMA-Sim
instructions, `prepare_case` builds the inputs and timed functions, `validate`
checks them, and `time_blocks` and `summarize` define timing and aggregation. In `plot_mmasim.py`, `COLORS` and
`PLOTS` define the presentation, `plot_points` selects the plotted data, and
`plot_results` contains the shared chart layout. The plotter imports no models.

To try a new plot style without rerunning or modifying an experiment:

```sh
./.venv/bin/python evaluation/performance/plot_mmasim.py evaluation/results/<run-id> \
  --output /tmp/fpy-plots
```
