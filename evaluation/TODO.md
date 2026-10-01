# MMA-Sim arithmetic-kernel benchmark

Benchmark the compiled FPy MMA arithmetic kernels against the eager PyTorch
implementations in the local MMA-Sim checkout at `~/repos/mma-sim`.

## Scope and decisions

- [x] Compare arithmetic kernels through the batched dot-product-accumulate
  (`dpa`) interface, not complete matrix-shaped `A * B + C` operations.
- [x] Use eager MMA-Sim as the first and only baseline. A `torch.compile`
  baseline is out of scope for the initial evaluation.
- [x] Produce the official results on the current Mac.
- [x] Keep the harness, documentation, raw measurements, metadata, summaries,
  and plots under `evaluation/`.
- [x] Benchmark the 14 designs currently exposed through `fpy2_models`:
  - NVIDIA Volta FP16/FP32
  - NVIDIA Turing FP16/FP32
  - NVIDIA Ampere TF32/FP32
  - NVIDIA Ampere BF16/FP32
  - NVIDIA Ada E5M2/FP32
  - NVIDIA Hopper FP16/FP32
  - NVIDIA Blackwell MXFP8
  - NVIDIA Blackwell NVFP4
  - AMD CDNA2 BF16
  - AMD CDNA2 FP16
  - AMD CDNA3 BF16
  - AMD CDNA3 FP16
  - AMD CDNA3 BF8
  - FP64 FMA reference
- [x] Report AMD CDNA1 BF16 and FP16 as unsupported. They remain excluded
  until the FPy C++ backend can compile their exponent-aligned FDPA models.

## 1. Benchmark layout

- [x] Add `evaluation/benchmark_mmasim.py` as the executable harness.
- [x] Add `evaluation/README.md` with setup and reproduction commands.
- [x] Write generated artifacts beneath `evaluation/results/<run-id>/` so
  separate runs do not overwrite one another.
- [x] Import MMA-Sim directly from `~/repos/mma-sim/src/mmasim` rather than
  relying on an independently installed wheel.
- [x] Fail with an actionable message when either the local MMA-Sim checkout
  or the built `fpy2_models` extension cannot be found.

## 2. Case registry

- [x] Define one declarative case for each design, recording:
  - display name and architecture
  - FPy raw `torch.ops.fpy2_models.*` operator
  - equivalent eager MMA-Sim arithmetic object and `dpa` call
  - operand, scale, accumulator, and output formats
  - fixed dot-product length `K`
  - scalar or grouped scale configuration where applicable
- [x] Keep input preparation separate from timed callables.
- [x] Use the same underlying quantized values for both implementations.
- [x] Pass storage-format tensors to MMA-Sim and the corresponding generated
  C++ compute-format tensors to the FPy operator.
- [x] Use a constant accumulator across a batch because the current FPy
  operator schema accepts scalar `c`, while MMA-Sim accepts a tensor.

## 3. Correctness gate

- [x] Before timing a case, run both implementations on its prepared inputs.
- [x] Compare output bit patterns, including NaNs and signed zero, rather than
  using an approximate tolerance.
- [x] Mark a case invalid and do not report its speedup when outputs differ.
- [x] Record enough information to reproduce a mismatch: case, batch size,
  workload, seed, formats, and first differing output.
- [x] Add a smoke test that exercises every registry entry with a small batch.

## 4. Workloads

- [x] Use deterministic seeds and generate source values only once per case.
- [x] Make finite normally distributed inputs the primary workload.
- [x] Add a secondary zero/subnormal-heavy workload to expose data-dependent
  behavior without mixing it into the headline results.
- [x] Quantize operands and scales before entering the timed region.
- [x] Sweep batch sizes `1`, `16`, `256`, `4096`, and `65536` initially.
- [x] Allow batch sizes, workloads, designs, seed, and minimum run time to be
  selected from the command line.

## 5. Timing protocol

- [x] Use `torch.utils.benchmark.Timer` and `blocked_autorange`.
- [x] Warm up imports, native extensions, allocators, and every timed callable.
- [x] Run the primary measurements with one Torch intra-op thread.
- [x] Optionally support a second run at the Mac's default Torch thread count,
  but keep it separate from the official single-thread comparison.
- [x] Collect several independent replicates for every configuration.
- [x] Alternate or randomize FPy and MMA-Sim measurement order to reduce
  thermal and order bias.
- [x] Keep model construction, input generation, quantization, validation,
  and compilation outside the timed region.
- [x] Record raw replicate measurements rather than only aggregate values.

## 6. Metrics and artifacts

- [x] Save long-form `measurements.csv` containing at least:
  - design, implementation, workload, batch size, and thread count
  - replicate and inner-loop counts
  - elapsed time, median, and interquartile range
  - nanoseconds per DPA and DPAs per second
- [x] Save `metadata.json` containing:
  - FPy and MMA-Sim Git revisions and dirty-worktree state
  - Python, PyTorch, and TorchAO versions
  - macOS version, CPU description, and logical CPU count
  - Torch intra-op and inter-op thread counts
  - native extension compiler and optimization flags
  - full benchmark command and timestamp
- [x] Print a compact table from the separate plotting script with per-design
  median throughput and FPy-over-MMA-Sim speedup.
- [x] Plot throughput versus batch size in a separate post-processing step.
- [x] Plot FPy-over-MMA-Sim speedup versus batch size in that step.
- [ ] Clearly label invalid, unsupported, and untimed cases in tables and
  plots rather than dropping them.

## 7. Reproduction and review

- [x] Document how to build `cpp_extension/fpy2_models` before benchmarking.
- [x] Document the exact command used for the official Mac run.
- [x] Run the registry smoke test and the existing extension tests.
- [x] Run a short benchmark pass to validate output generation.
- [x] Run the full official benchmark with the Mac connected to power and
  otherwise idle.
- [x] Review raw distributions for unstable cases before interpreting medians.
- [x] Summarize the main crossover points, throughput differences, and any
  workload-sensitive results without extrapolating to GPU performance.

## Deferred work

- [ ] Add a `torch.compile` MMA-Sim baseline if eager results motivate it.
- [ ] Add full matrix-shaped `A * B + C` measurements after the FPy adapter
  supports equivalent per-output accumulators.
- [ ] Add CDNA1 after those designs compile through the C++ backend.
- [ ] Repeat on other CPU systems only as a separate, explicitly identified
  evaluation.
