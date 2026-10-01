# Apple M3 Pro results: eager MMA-Sim and unsafe-array FPy harness

This report summarizes the official single-thread Mac run after removing the
per-row array copies from the FPy C++ Torch harness. The generated result
directory is intentionally not stored in Git; rerunning the documented
benchmark produces metadata, raw measurements, validation records, and plots.

## Compared paths

- `fpy-core`: the raw compiled operator, given prequantized values in its C++
  compute type. This is the primary arithmetic-kernel comparison.
- `fpy-storage`: the public FPy wrapper, given storage-format inputs. This
  includes storage-to-compute adaptation inside the timed call.
- `mmasim`: the equivalent eager MMA-Sim `dpa` arithmetic model, given
  storage-format inputs.

Every timed configuration passed bitwise comparison, including signed zero
and NaN representations: 14 designs, two workloads, and five batch sizes, for
140 validated configurations. CDNA1 BF16 and FP16 remain unsupported because
those FPy models do not compile to C++.

## Headline result

For finite normal inputs at batch size 65,536, `fpy-core` was faster than
eager MMA-Sim for all 14 designs. The median speedup across designs was 1.61x;
the range was 1.11x to 5.15x.

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

The Ada E5M2 eager MMA-Sim headline series had an 11.3% relative IQR and is
flagged in the console output. The other batch-65,536 normal headline series
were below the 10% instability threshold.

## Interpretation

The eager MMA-Sim implementation is tensorized, so its dispatch overhead is
amortized as the batch grows. This is visible in the narrowing speedup: across
normal workloads, the median `fpy-core` speedup falls from 88.61x at batch 1
to 1.61x at batch 65,536. MMA-Sim still performs many separately dispatched
full-tensor operations and creates intermediate tensors, while the FPy path
runs one fused native model for each row.

### Why fused FPy can beat tensorized MMA-Sim

Tensorization and fusion are different optimizations. Eager MMA-Sim applies
each PyTorch primitive to the complete batch, but primitives such as `frexp`,
comparison, masking, concatenation, exponentiation, reduction, and conversion
remain separate dispatches. Intermediate tensors are written and subsequently
read by later primitives. Tensorization amortizes Python and dispatch overhead
and lets each primitive use an efficient loop, but it does not combine the
complete arithmetic model into one loop.

FPy instead generates one specialized C++ function for a fixed design and
fixed dot-product length. The Torch operator enters native code once, walks the
batch, and executes the complete model for each row. Most intermediate values
remain scalar or fixed-size local values rather than materialized PyTorch
tensors. `-O3` compilation can simplify fixed trip counts, propagate constants,
and keep short-lived values in registers or cache. The unsafe array view also
lets each model read a contiguous tensor row directly instead of copying it
into a temporary top-level `std::array`.

An illustrative profile of Ampere BF16 at batch 65,536 showed this structural
difference. The FPy side appeared as one native operator and allocated its
256 KiB output. The eager MMA-Sim call involved 43 PyTorch operator categories;
summing positive per-operator allocation events gave roughly 129 MiB of gross
temporary allocation traffic. That gross value is not peak live memory, but it
does show why a tensorized sequence can remain memory- and allocation-heavy.

The scaling curve supports this explanation. At batch 1, where eager dispatch
and allocation dominate, the median FPy advantage across designs was 88.61x.
At batch 65,536, MMA-Sim amortized that overhead and the median gap narrowed to
1.61x. Ampere TF32 was only 1.11x faster in `fpy-core`, while the largest gap
was 5.15x for the very small fixed-size FP64 FMA model.

### Limitations and comparability caveats

1. **The baseline is eager PyTorch, not compiled PyTorch.** `torch.compile`
   could fuse some MMA-Sim operations, remove intermediates, or specialize for
   the fixed shapes. It may also encounter graph breaks in these models. This
   benchmark says nothing about that currently unmeasured baseline. Here,
   "compiled MMA-Sim" would mean wrapping each MMA-Sim `dpa` method with
   `torch.compile`, triggering compilation and warm-up before timing, and then
   measuring its steady-state calls as a separate `mmasim-compiled` series.
   This would be a comparison between two compilation approaches, not
   necessarily two single native kernels: tensor-dependent Python branches in
   MMA-Sim can cause graph breaks and leave portions of a call executing
   eagerly. Any compilation failures and graph breaks would need to be
   reported explicitly rather than silently falling back to the eager result.
   Consequently, the present numbers demonstrate the benefit of FPy's
   specialization and fusion over MMA-Sim's eager execution path; they do not
   establish that FPy is faster than a compiled MMA-Sim implementation.

2. **This is a CPU-only, single-thread result.** Measurements used one Torch
   and OpenMP thread on one Apple M3 Pro. MMA-Sim can run on GPUs and may scale
   differently with CPU threads. FPy's current operator is CPU-only and its
   outer batch loop is serial. No conclusion should be extrapolated to GPU
   execution or another CPU architecture.

3. **The primary interfaces use different physical input representations.**
   `fpy-core` receives already-quantized values widened to the generated C++
   compute type, normally FP32. MMA-Sim receives storage-typed tensors and
   performs the normalization and conversions required by its implementation.
   The `fpy-storage` series includes FPy's corresponding input adaptation and
   therefore provides a useful second bound, but it is not structurally
   identical to MMA-Sim's input path.

   A focused Ampere BF16 audit found that this distinction did not explain its
   result: at batch 65,536, `fpy-core`, `fpy-storage`, and eager MMA-Sim took
   approximately 12.15 ms, 12.52 ms, and 14.22 ms respectively. Giving
   MMA-Sim the same widened values took approximately 14.76 ms. This check is
   reassuring for that design, but it is not proof for every format.

4. **The accumulator interfaces differ.** FPy currently accepts one scalar
   `c` for the entire batch. MMA-Sim accepts a tensor accumulator, so the
   benchmark gives it a batch-sized tensor filled with the same scalar. A
   focused zero-stride expanded-accumulator test did not materially change the
   Ampere BF16 result, but FPy still performs less accumulator input traffic.

5. **NVFP4 adapter timing is deliberately asymmetric.** There is no native
   Torch FP4 tensor dtype in this path. Both arithmetic kernels receive
   unpacked, already-quantized E2M1 values, while `fpy-storage` invokes TorchAO
   quantization again inside the timed public wrapper. Its 0.09x headline
   result measures that adapter cost and should not be compared with the raw
   arithmetic-kernel ratio of 1.33x.

6. **These are DPA microbenchmarks, not full matrix operations.** The harness
   calls the arithmetic `dpa` models on independent vectors. It excludes
   matrix expansion, layout transformations, packing, and per-output
   accumulators from a complete `A * B + C` interface. The fixed `K` values are
   those accepted by the current generated wrappers and should not be treated
   as an end-to-end GEMM workload.

7. **Different designs do different amounts of work.** Dot-product lengths and
   arithmetic models vary. DPA/s is meaningful for comparing FPy with MMA-Sim
   within one design, but raw throughput should not be used to rank different
   designs against each other.

8. **Inputs cover only two synthetic distributions.** The primary workload is
   finite normal data; the second is zero/subnormal-heavy. Both use a constant
   accumulator and deterministic inputs. Special values and production input
   distributions could change branch behavior and relative throughput.

9. **Correctness validation is configuration-specific.** Every timed input in
   this run matched bit-for-bit, which prevents timing two observably different
   computations on those inputs. It is not an exhaustive proof that the two
   implementations agree for every possible bit pattern.

10. **The unsafe array view is non-portable C++.** Compile-time assertions
    verify the current `std::array` size and alignment, but tensor storage does
    not formally contain live `std::array` objects under the C++ object model.
    A compiler or standard-library change could invalidate the assumption. A
    generated `std::span` or pointer entry interface would provide a safer
    zero-copy comparison.

11. **The run is subject to system noise.** macOS CPU affinity was not pinned,
    and an M3 Pro has heterogeneous performance and efficiency cores. Thermal,
    power, and scheduler effects remain possible. Three interleaved replicates
    and autoranged samples reduce order bias, but 17 of 420 timing series had a
    relative IQR above 10%; the affected headline Ada E5M2 MMA-Sim series is
    explicitly flagged.

12. **This run does not isolate the array-cast optimization.** It measures the
    finished unsafe-array implementation against MMA-Sim. The earlier copying
    run used a less controlled timing protocol and contained a clear Ampere
    BF16 outlier, so comparing the two result directories would not be a clean
    A/B estimate of the copies' cost.

## Generated artifacts

- `measurements.csv`: normalized raw timing blocks and aggregates
- `summary.csv`: one aggregate row per timed series
- `validation.json`: correctness results and unsupported designs
- `metadata.json`: complete run provenance
- `throughput-normal.png`, `throughput-stress.png`: throughput scaling
- `speedup-normal.png`, `speedup-stress.png`: relative scaling

These files live under `evaluation/results/<run-id>/` and are ignored by Git.
