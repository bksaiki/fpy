# Fixed-length dot-product evaluation

Compare compiled FPy models against the equivalent eager MMA-Sim models by
computing complete dot products of lengths **1024, 2048, and 8192**, with batches
of **1, 16, and 256** independent dot products by default.

For each design, split each vector into consecutive chunks of the model's
existing width (4–64 terms). Start an accumulator at zero, call the unchanged
DPA model on each chunk with the previous result as its accumulator, and return
the final result. Each model retains its own arithmetic and rounding semantics.
Different designs need not produce the same answer; each FPy design must match
its corresponding MMA-Sim design bit for bit before it is timed.

FPy runs the chunk loop in native C++, calling the existing generated model.
MMA-Sim runs its existing eager batched `dpa` sequentially over chunks. A batch
of 16 means 16 independent length-L dot products with separate accumulators.
Batch 1 emphasizes latency; larger batches amortize MMA-Sim dispatch overhead.
The comparison measures compiled FPy versus eager MMA-Sim, not `torch.compile`
or GPU performance. The native FPy loop is serial; the default is one Torch
thread. Optional thread counts affect Torch/MMA-Sim rather than parallelizing
the FPy wrapper.

There are two series:

- `fpy-core`: prepared compute-format inputs passed to the compiled dot wrapper.
- `mmasim`: equivalent storage-format inputs passed to eager MMA-Sim.

Input generation, quantization, chunk preparation, compilation, and correctness
validation are outside timing. Both paths use the same quantized numerical
operand values, in the physical representations their interfaces require.
MMA-Sim chunk tensors are made contiguous ahead of timing. Timings include
execution, dispatch, accumulator initialization, and output allocation. Every
invocation starts fresh at zero. The former `fpy-quantize` series is removed.

Inputs are sampled from N(0, 0.5²) in FP64 and rounded to each design's input
format. MXFP8 and NVFP4 operands are block-quantized with TorchAO: one E8M0
scale per 32 elements (`to_mx`) and one E4M3 scale per 16 elements
(single-level `nvfp4_quantize`; the per-tensor scale lies outside the MMA).
Every chunk receives its own block scales.

Each MMA-Sim model comes from MMA-Sim's instruction simulators (for example
`nv_ptx.MMA("Volta", "m8n8k4.f32.f16.f16.f32")`), which select the arithmetic
model and its parameters; `validation.json` records the instruction for each
design. CDNA1 remains unsupported by the C++ backend.

## Build and run

Build from the repository root:

```sh
(cd cpp_extension && ../.venv/bin/python setup.py build_ext --inplace)
./.venv/bin/python -m pytest evaluation/test_benchmark_mmasim.py cpp_extension/tests -q
```

The build command runs `examples/mmasim/compile.py` to generate C++ sources
under `cpp_extension/build/generated_models/`, then compiles and links those
sources and the Torch wrappers with `-O3`. Benchmarking and plotting do not
rebuild the FPy models.

The benchmark uses `~/repos/mma-sim` by default; override with
`--mmasim-root PATH`. On macOS it selects Homebrew LLVM when available for
MMA-Sim's OpenMP helper unless `CXX` is already set.

Run the full evaluation:

```sh
./.venv/bin/python evaluation/benchmark_mmasim.py \
  --dot-lengths 1024,2048,8192 --batch-sizes 1,16,256 \
  --threads 1 --replicates 3 --min-run-time 0.25
```

For a short smoke run, add `--design amd.cdna2.f16 --replicates 1
--min-run-time 0.05`. Use `--validate-only` for correctness
without timings. Custom lengths must be positive multiples of every selected
model's width; partial chunks are rejected.

Each run writes a new directory under `evaluation/results/`:

- `metadata.json`: revisions, environment, timing parameters, lengths and batches.
- `validation.json`: bitwise correctness results for each configuration.
- `measurements.csv`: normalized timing samples and configuration aggregates.
- `summary.csv`: medians, IQRs, nanoseconds per complete dot product, complete
  dot products/second, and product terms/second.

Length, model width, and chunks per dot are recorded with timing results.
One timed call computes the entire batch; per-dot metrics divide by batch size.
No speedup is reported for a configuration that fails validation.

Generate throughput and speedup plots for each length, with one panel per batch
size:

```sh
./.venv/bin/python evaluation/plot_mmasim.py evaluation/results/<run-id>
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

Data collection and plotting are independent commands. Edit `build_cases` and
`prepare_case` in `benchmark_mmasim.py` to change the designs or inputs; `time_callable`
and `summarize` define timing and aggregation. In `plot_mmasim.py`, `COLORS` and
`PLOTS` define the presentation, `plot_points` selects the plotted data, and
`plot_results` contains the shared chart layout. The plotter imports no models.

To try a new plot style without rerunning or modifying an experiment:

```sh
./.venv/bin/python evaluation/plot_mmasim.py evaluation/results/<run-id> \
  --output /tmp/fpy-plots
```
