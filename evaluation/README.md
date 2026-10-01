# MMA-Sim arithmetic-kernel evaluation

This directory compares the compiled FPy dot-product-accumulate kernels with
the equivalent eager PyTorch arithmetic in a local MMA-Sim checkout. It does
not benchmark complete matrix-shaped MMA calls or `torch.compile`.

The correctness gate requires bitwise-identical outputs before a configuration
is timed. Input generation, model construction, compilation, and validation
are outside the timed region. Each configuration measures three paths:

- `fpy-core`: prequantized values passed directly to the compiled operator
- `fpy-storage`: storage-format values passed through the public FPy adapter
- `mmasim`: storage-format values passed to eager MMA-Sim `dpa`

Quantization is outside the arithmetic-only `fpy-core` and MMA-Sim comparison.
The storage path times the FPy adapter's conversions and, for NVFP4, its
TorchAO quantization.

## Results

The [reported run](RESULTS.md) used one thread on an 11-core Apple M3 Pro with
PyTorch 2.14.0 and eager MMA-Sim at revision `aab7f2a`. All 140
configurations—14 designs, two input workloads, and five batch sizes—passed
bitwise validation for both FPy paths. CDNA1 BF16 and FP16 were not timed
because those FPy models do not yet compile to C++.

For normal inputs at batch size 65,536, `fpy-core` was faster than eager
MMA-Sim for all 14 designs. Its median speedup across designs was **1.61x**,
with a range of **1.11x to 5.15x**. The storage-input FPy path had a median
speedup of **1.57x**; NVFP4 is the exception because its public wrapper repeats
TorchAO E2M1 quantization inside the timed call.

| Design | FPy core M DPA/s | FPy storage M DPA/s | MMA-Sim M DPA/s | Core speedup | Storage speedup |
|---|---:|---:|---:|---:|---:|
| `nv.volta.f16.f32` | 7.11 | 7.16 | 5.34 | 1.33x | 1.34x |
| `nv.turing.f16.f32` | 5.26 | 5.26 | 3.54 | 1.49x | 1.48x |
| `nv.ampere.tf32.f32` | 7.81 | 7.59 | 7.05 | 1.11x | 1.08x |
| `nv.ampere.bf16.f32` | 5.29 | 4.96 | 4.54 | 1.17x | 1.09x |
| `nv.ada.e5m2.f32` | 7.54 | 7.02 | 3.68 | 2.05x | 1.91x |
| `nv.hopper.f16.f32` | 3.09 | 3.02 | 1.86 | 1.66x | 1.62x |
| `nv.blackwell.mxfp8` | 2.78 | 2.69 | 1.89 | 1.47x | 1.43x |
| `nv.blackwell.nvfp4` | 9.62 | 0.69 | 7.24 | 1.33x | 0.09x |
| `amd.cdna2.bf16` | 246.85 | 199.80 | 59.50 | 4.15x | 3.36x |
| `amd.cdna2.f16` | 311.15 | 241.66 | 66.45 | 4.68x | 3.64x |
| `amd.cdna3.f16` | 14.35 | 13.98 | 7.07 | 2.03x | 1.98x |
| `amd.cdna3.bf16` | 14.08 | 13.73 | 9.02 | 1.56x | 1.52x |
| `amd.cdna3.bf8` | 6.38 | 5.59 | 3.27 | 1.95x | 1.71x |
| `fp64.fma` | 1197.90 | 1174.81 | 232.40 | 5.15x | 5.06x |

Tensorization does not fuse MMA-Sim's full arithmetic model: eager execution
still dispatches many whole-tensor primitives and materializes intermediate
tensors. FPy enters one fixed-size, `-O3`-compiled native operator and keeps
most intermediates local. The advantage consequently narrows as MMA-Sim
amortizes dispatch costs: the median normal-input speedup falls from 88.61x at
batch 1 to 1.61x at batch 65,536.

These results are limited to eager, single-threaded CPU DPA microbenchmarks on
this Mac. They do not cover `torch.compile`, GPUs, multithread scaling, or full
matrix operations. The two primary interfaces also use different physical
input representations, FPy uses a scalar accumulator, and the unsafe
`std::array` view is not portable under the strict C++ object model. System
noise remains visible: 17 of 420 timing series exceeded a 10% relative IQR,
including the headline Ada E5M2 MMA-Sim series. See the
[`full results report`](RESULTS.md) for
the complete explanation, limitations, raw-data links, and profiling evidence.

In particular, the comparison is compiled FPy versus **eager** MMA-Sim. A
useful follow-up would pass MMA-Sim's `dpa` methods through `torch.compile`,
exclude compilation and warm-up from the timed region, and report that as a
separate `mmasim-compiled` series. Such a series would test whether FPy's
advantage remains after PyTorch has an opportunity to fuse and specialize
MMA-Sim's tensor operations. It would not necessarily be a single fully
compiled kernel: MMA-Sim contains tensor-dependent Python branches and other
constructs that can cause graph breaks, leaving portions in eager execution.
Compilation failures and graph breaks would therefore need to be reported
rather than silently replaced with eager timings. Until that baseline is
measured, the results demonstrate the advantage of FPy's specialization and
fusion over MMA-Sim's current eager execution path, not an advantage over a
compiled MMA-Sim implementation.

## Setup

From the repository root, build the FPy Torch extension in place:

```sh
cd cpp_extension
../.venv/bin/python setup.py build_ext --inplace
cd ..
```

The build regenerates the per-design C++ translation units under the ignored
`cpp_extension/build/generated_models/` directory. Generated model sources and
native binaries are not stored in Git.

The benchmark defaults to the MMA-Sim checkout at `~/repos/mma-sim`. Select a
different checkout with `--mmasim-root PATH`.

MMA-Sim's FP64 CPU helper uses OpenMP. On this Mac, the harness automatically
uses `/opt/homebrew/opt/llvm/bin/clang++` when `CXX` is unset because Apple
Clang rejects MMA-Sim's `-fopenmp` option. The chosen compiler is saved in the
run metadata.

## Validation and short run

Run the registry smoke tests:

```sh
./.venv/bin/python -m pytest evaluation/test_benchmark_mmasim.py -q
```

Run a short timing pass:

```sh
./.venv/bin/python evaluation/benchmark_mmasim.py \
  --workloads normal --batch-sizes 1,16 --replicates 1 --min-run-time 0.05
```

The benchmark only collects and summarizes measurements. To print the headline
table and generate plots afterward, pass the result directory printed by the
benchmark to the plotting script:

```sh
./.venv/bin/python evaluation/plot_mmasim.py evaluation/results/<run-id>
```

Use `--design SUBSTRING` one or more times to select designs. Use
`--validate-only` to exercise the correctness gate without collecting timing
samples.

## Official Mac run

Connect the Mac to power, close compute-heavy applications, and run:

```sh
./.venv/bin/python evaluation/benchmark_mmasim.py \
  --threads 1 \
  --batch-sizes 1,16,256,4096,65536 \
  --workloads normal,stress \
  --replicates 3 \
  --min-run-time 0.25
```

Each run creates a new timestamped directory beneath the Git-ignored
`evaluation/results/` directory containing:

- `metadata.json`: revisions, dirty state, software and hardware versions,
  compiler information, and benchmark parameters
- `validation.json`: correctness status and mismatch details
- `measurements.csv`: normalized raw timing blocks
- `summary.csv`: median, interquartile range, time per DPA, and throughput

Running `plot_mmasim.py` adds `throughput-*.png` and `speedup-*.png` to that
directory and prints the headline table. Use `--workload` or `--batch-size` to
select a different headline while retaining plots for every measured workload;
use `--output` to write plots elsewhere.

The reported core and storage speedups divide the corresponding FPy throughput
by eager MMA-Sim throughput, so values greater than one favor FPy. A `!` in
the console summary marks a series whose interquartile range exceeds 10% of
its median. Results describe CPU simulation performance on the recorded Mac
and should not be extrapolated to GPU execution.
