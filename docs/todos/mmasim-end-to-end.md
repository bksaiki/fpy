# mmasim end-to-end: tractable evaluations through seeded subsets

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.  Code under `examples/mmasim/serve/`; the
parent plan is `docs/todos/mmasim-serving.md` (Nice to have: "Tractable
end-to-end evaluations").

## Working policy

- **Pause after each phase for review.** Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.** The author of the change leaves the working tree dirty;
  commits are made by the repository owner.
- **Run only the tests relevant to the phase.** The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* -- what was tried, what
  a phase decided, why an ordering was chosen -- belong in this document, not in
  source comments.

## Context

Local metrics are the primary evaluation.  The end-to-end ones confirm it on a
few designs, but they are too slow to do even that:

| script | cost per design (Qwen3-0.6B, TITAN V) |
|---|---|
| `perplexity.py`, 146 segments | ~1.6 h for the full pass |
| `zeroshot.py`, full suite | hours |
| `decode.py`, 100 prompts | hours: launch-bound at `m = 1`, and most prompts never diverge |

Each script already has a size knob, but they don't agree:

| script | knob | how it picks | uncertainty reported |
|---|---|---|---|
| `perplexity.py` | `--segments N` | the first N | per run, over tokens (llama.cpp's) |
| `zeroshot.py` | `--hellaswag N`, `--limit N` | HellaSwag seeded (`Random(0)`); `--limit` the first N per task | per run, the harness's |
| `decode.py` | `--prompts N`, `--max-new` | seeded (`Random(0)`) | none |
| `local.py` | `--tokens N` | first N (WikiText), seeded `Random(0)` (MT-Bench) | none |

What is wrong with that:
- **"First N" is biased:** it takes a contiguous stretch of WikiText-2's
  articles.
- **The seed is hard-coded,** so a result can't be checked on a second
  subset.
- **The standard errors are per run, not paired.** Every run sees the same
  inputs, so the quantity that matters is a design's *difference* from a
  reference, and its error bars are far tighter than two runs' separate ones.
  With independent error bars, a 20-segment result can't separate designs
  whose KLs differ by ~5%.
- **Standard errors are over tokens,** which are not independent, and so
  understate the uncertainty.
- **`perplexity.py`, `zeroshot.py` and `decode.py` run only `bf16`,** so the
  quantized designs have no end-to-end numbers.  That blocks the parent
  plan's essential gap, "local metrics predict end-to-end effects".
- **`layers.py` contributes little.**
  - Its local metrics repeat `local.py`'s on each run's own inputs.  Under
    `bf16` they agree within ~0.01 in log2, which is `test_local.py`'s
    cross-check.
  - Its only metric of its own, `propagated`, asks what end-to-end KL
    answers once KL is affordable.
  - It hosts the metric code `local.py` imports (`Stats`, `local`, `fmt`),
    by accident of history.

## Design

### Seeded subsets

- **One helper,** `swap.pick(total, n, seed) -> list[int]`: `n`
  indices of `range(total)`, sorted, from `random.Random(seed)`; all of them
  when `n` is `None` or `>= total`.  `local.sample` uses it.
- **Size flags keep their units:** `--segments` (perplexity), `--items` (zero
  shot, per task), `--prompts` (decode), `--tokens` (local).  **One shared
  `--seed`, default 0**, added by `swap.add_args`.
- **Every selection is random:** "first N" is gone.  `zeroshot.py`'s
  `--hellaswag` and `--limit` become `--items`, a cap applied to every task.
- **Recorded:** every JSON holds the seed and the sizes.  The caches of
  `zeroshot.py` and `decode.py` already recompute on any settings mismatch,
  so adding the seed to `settings` is enough to keep pairs from mixing.

### Paired statistics

Every run sees the same subset, so each run is reported against a
reference, with standard errors over the sampling unit rather than tokens:

| script | unit | per unit | reported |
|---|---|---|---|
| perplexity | segment | mean NLL, KL, top-1 disagreement | Δ NLL (log of the PPL ratio), KL, top-1 disagreement; mean and SE over segments |
| zeroshot | item | correctness `c` | Δ acc = mean(`c_run - c_ref`), SE `sd(c_run - c_ref) / sqrt(n)`; flips as now |
| decode | prompt | divergence index (censored at R0's length), disagreement rate, KL | fraction diverged, mean disagreement and KL; SE over prompts |

- **Two references** (`mmasim-serving.md`, Runs):
  - R0, the pre-quantization reference, for the observable effect;
  - R1, `<scheme>-exact`, the post-quantization reference, for the
    design's own effect, which is what local metrics predict.

  Under `bf16`, R1 is `bf16-exact`.
- The per-run, token-level numbers stay (PPL, llama.cpp's KL and top-1),
  for comparison with published figures.

### Schemes end to end

- **`--scheme`, `--requantize` and `--master`** go to `perplexity.py`,
  `zeroshot.py` and `decode.py`, as `local.py` has them.
- **`--model`** takes a master (RTN) or a checkpoint.
- **`local._load` moves to `checkpoints.for_scheme`,** so that every script
  loads a model and its `Run` in one way.  R0 is the model's FP32 weights:
  a checkpoint's dequantized ones.
- **`-r` choices** are `swap.modes(scheme)`, checked after parsing.

### Teacher-forced divergence

The free-running divergence index costs a full decode per design.  The
replacement:
- R0 still decodes greedily, once, and is cached.
- Each design then runs one prefill over prompt plus R0's output.
- At each generated position, compare the design's argmax with R0's argmax
  on the same forced sequence: R0 also prefilled, so that both come from
  the same computation path.

Up to the first disagreement the prefixes are identical, so this gives the
same index as free running, except where prefill and incremental decode
compute attention differently (measured in Phase 5: at near-ties, often).
It also gives what free running cannot: disagreement and KL at every later
position.  The cost per design drops to about one perplexity segment per
prompt.  Logits go through in blocks of
positions, as `perplexity._log_probs` does.

## Phases

### Phase 1 -- Retire `layers.py`

**Done.**  As planned; `local.py`'s output on the regression net (512
WikiText-2 tokens, `bf16`) is identical before and after, every layer and
total.  One fix on the way: `local.evaluate`'s `metrics` parameter and a
variable in `main` would have shadowed the module, and are now `names`.

- **What:**
  - `Stats`, `_against`, `local`, `fmt`, `U` and `QUANTIZED_ONLY` move to
    `serve/metrics.py`, with `propagated` dropped from `METRICS` and
    `Stats`.
  - `layers.py` and `tests/test_layers.py` are deleted.
  - `test_local.py`'s cross-check gets its own small hook: the model run
    through a design, local metrics on its own inputs.
  - `mmasim-serving.md`: layout, interface and the metrics table's
    `propagated` row.  Its past finding stays in the record.
- **Why first:** later phases add statistics that `local.py` shares (Phase
  4, and a later bootstrap over sequences), and they should land in
  `metrics.py`, not in a module about to go.
- **Regression net:** `local.py` on Qwen3-0.6B, `bf16`, 512 WikiText-2
  tokens, bit for bit before and after.
- **Tests:**

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_local.py serve/tests/test_checkpoints.py

### Phase 2 -- Seeded subsets

**Done.**  Where it departed:
- **`pick` is in `swap.py`,** next to `add_args`, not in `workloads.py`,
  which imports `perplexity` and `decode` and so would make an import cycle.
- **`add_args(ap, seed=False)` for `chat.py`,** which has nothing to sample.
- **No cache test:** the caches compare `settings` for equality, and the
  seed is in `settings`, so a helper just to test it would be code for its
  own sake.
- **`local.py`'s WikiText workload stays the first 2048 tokens:** one
  sequence, all of it sampled, so `--seed` changes nothing there.  A random
  window would change every WikiText result in `mmasim-serving.md`.
- **The tests** for `pick` and the zero-shot subsets are in
  `test_zeroshot.py`, which needs no GPU.
- **The harness's counts** match the Open items': PIQA 1,838, ARC-e 2,376,
  ARC-c 1,172, HellaSwag 10,042, WinoGrande 1,267, LAMBADA 5,153.  At 2,000,
  ARC-e, HellaSwag and LAMBADA are subsampled.

- **What:**
  - `swap.pick`; `--seed` in `swap.add_args`.
  - `local.sample(seqs, tokens, seed)`.
  - `perplexity.py --segments` picks at random.
  - `zeroshot.py --items` replaces `--hellaswag` / `--limit`, and passes
    `samples=` for every task.
  - `decode.py --prompts` uses `pick`.
  - The seed is in every JSON and every cache's `settings`.
- **Why first:** everything later reports on a subset, and this fixes what
  the subset is.
- **Regression net:** at seed 0, `local.py` picks what it picks today
  (`Random(0)` on the same range), and `zeroshot.py`'s HellaSwag subset is
  today's.  Checked in the test, not assumed.
- **Tests:**
  - `pick`: fixed per seed; sorted; everything at `n >= total`; another seed
    gives another pick.
  - `local.sample` at seed 0 unchanged.
  - A zeroshot cache with another seed is recomputed; unit, no harness.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_workloads.py serve/tests/test_local.py serve/tests/test_zeroshot.py

### Phase 3 -- Schemes end to end

**Done.**  Where it departed:
- **`for_scheme` returns `(model, run, about)`** and loads no master
  weights.  End to end, R0 is the model itself; `local.py` loads the master
  where it needs one.
- **`checkpoints.runs(ap, args)`** checks `-r` against the scheme's runs,
  since argparse's `choices` can't depend on `--scheme`.  `swap.RUNS` went
  with it.
- **The caches key R0 without the scheme,** so one R0 serves every scheme.
  A design's settings add the scheme and `--requantize`.
- **No `for_scheme` test,** because it needs the Hub.  Instead, a smoke run:
  `perplexity.py --scheme fp8-row --segments 1` on Qwen3-0.6B.  It gives KL
  vs R0 of 1.95e-2 for `fp8-row-exact` and 1.93e-2 for Blackwell, which is
  the quantization's effect, as local metrics found.
- **The test's "each design nearer `fp8-row-exact`"** waits for Phase 4's
  second reference.  For now, every design runs with a finite, nonzero KL.

- **What:**
  - `checkpoints.for_scheme(name, scheme, *, requantize, master, split_k,
    combine)`, moved from `local._load`.
  - Shared flags `--scheme`, `--requantize`, `--master` in
    `checkpoints.add_args`.
  - `perplexity.py`, `zeroshot.py` and `decode.py` run `swap.modes(scheme)`,
    with `lm_head` ignored under a quantizing scheme, as in `local.py`.
- **Why separate:** it changes how every script loads a model, independent
  of what the scripts report.
- **Tests:**
  - `perplexity.evaluate` under `fp8-row` on the test model: `fp8-row-exact`
    is farther from R0 than `bf16-exact`, and each design is nearer
    `fp8-row-exact` than R0 is.
  - `local`'s tests pass through `for_scheme`.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_perplexity.py serve/tests/test_local.py serve/tests/test_checkpoints.py

### Phase 4 -- Paired statistics

**Done.**  Notes:
- **`metrics.paired`, `p_value` and `holm`** hold the statistics.  The
  family for Holm is every run compared with the same reference.
- **Per-segment values go into the JSON** (`per_segment`: NLL, and KL and
  disagreement per reference), so any other pairing can be done offline:
  design vs design, for one.
- **The exact run always goes right after R0,** whatever `-r`'s order, so
  that every design has both references.
- **Zero-shot writes `<out>/paired.json`.**  A metric that no task logs is
  left out of the macro average.
- **Smoke run** (Qwen3-0.6B, `bf16`, 4 random segments):
  - Δ NLL's paired SE is ~2e-4, about 200x tighter than the per-run
    perplexity error bars (±0.88 on PPL, ≈0.04 in NLL).
  - No design's Δ NLL is significant against either reference (Holm
    p ≥ 0.5).
  - KL does not yet separate CDNA2 (4.73e-5 ±1.5e-6) from CDNA3 (4.79e-5
    ±1.8e-6).
  - So Phase 6's power analysis should be on design-vs-design differences
    in per-segment KL, not on tests of Δ NLL against zero.
  - 20 items per task through the zero-shot suite ran end to end.

- **What:**
  - `perplexity.Totals` keeps per-segment sums, and reports each run against
    both references, with SEs over segments.
  - `zeroshot` reports Δ acc with its paired SE, per task and macro-averaged,
    against both references.
  - The statistics live in `metrics.py`: a paired mean and SE over units,
    and Holm's correction for the designs compared in one run.
- **Why separate:** it only changes reporting, on data the earlier phases
  already produce.
- **Tests:**
  - A run against itself: Δ = 0, SE 0.
  - The paired SE equals a hand computation on made-up per-segment and
    per-item values.
  - The paired SE is below the two runs' separate SEs combined, on the test
    model.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_perplexity.py serve/tests/test_zeroshot.py

### Phase 5 -- Teacher-forced divergence

**Done.**  Where it departed:
- **No `decode.forced`.**  A prompt and R0's reply is a sequence scored from
  the reply's start, so `perplexity.evaluate` does it: it takes `starts`,
  and records each segment's first disagreement per reference and its
  first miss of the target.  The references, the pairing and Phase 4's
  statistics come with it.
- **`decode.py` caches only R0's tokens.**  A run's forced pass costs about
  one perplexity segment per prompt, so it isn't cached.  The results go to
  `<out>/forced.json`.
- **`greedy(ref=...)` and `divergence` stay** for running free.  The CLI
  has no free-running mode.
- **The test model never diverges:** R0's reply repeats one token with a
  top-2 margin ≥ 0.3, under every scheme.  So the tests check the index
  arithmetic, with a change planted at a known position, rather than
  free-vs-forced equality.

**Measured, Qwen3-0.6B, the 5 seed-0 prompts** (those of the old
free-running smoke run, which re-ran identically).  R0 is 2m20s, forced on
every run in the same call.
- **R0 forced misses none of its own tokens.**
- **The designs do not match** (divergence indices against R0):

  | run | free running | forced |
  |---|---|---|
  | bf16-exact | 607, 225 | 607, 225 |
  | amd.cdna2.bf16 | 713 | 368, 607, 225 |
  | nv.hopper.bf16.f32 | 607 | 420, 607 |

- **The kernels are not the cause:** every design's rows are bit-identical
  at `m = 1`, 7 and 100.
- **The attention path is the cause.**  Prefill and decoding round
  attention differently (FP32, torch), which moves the hidden states by
  ~1e-7.  At a design's near-tie, that decides the argmax.
- **Each method is self-consistent** (the design and R0 on one path), so
  both measure the design's divergence.  They agree in kind, but not prompt
  by prompt, and R0's clean floor does not bound the designs.
- **The docstring's equivalence claim is corrected.**  Whether the two
  agree in *rate* is open (below).

- **What:**
  - `decode.forced(model, prompt, ref)`: argmax and log-probabilities at each
    of *ref*'s positions, in one prefill.
  - `decode.py` compares every run with R0's forced argmax.  It reports the
    divergence index, disagreement rate and KL, with the Phase 4 paired SEs;
    the median index, not a mean, with a paired bootstrap over prompts.
  - `greedy` remains for R0's decode.
- **Why separate:** it replaces a measurement, so its equivalence with the
  old one is checked on its own.
- **Measurement to record:** how often R0's forced argmax differs from its
  own greedy tokens.  Prefill and incremental decode compute attention
  differently, so this is the floor of the method.
- **Tests:**
  - R0 forced against itself never diverges.
  - On the test model, where a design's free-running decode diverges, the
    forced index is the same.
  - Settings with a different R0 are recomputed.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_decode.py

### Phase 6 -- Sizes that suffice

**Done**, cut down from the plan: 30 random units per evaluation, not full
passes, since power analysis needs only the per-unit spread, which 30 units
estimate to ±13%.  Zero-shot was not run.  `results/phase6/`, 41 min on the
TITAN V.

Units for 80% power (two-sided, α = 0.05) to separate the widest pairs.
Each is from the per-unit difference between the two designs,
`n = (2.8 sd / mean)²`:

| evaluation (unit) | statistic | Ampere vs CDNA3 (`bf16`) | Ada vs Blackwell (`fp8-row`) |
|---|---|---|---|
| perplexity (segment) | KL vs R1 | **39** | **2** |
| | top-1 disagreement vs R1 | 95 | 62 |
| | KL vs R0 | 5,391 | 46 |
| | NLL | 58 | 433 |
| decode, forced (prompt) | KL vs R1 | 20,139 | 44 |
| | top-1 disagreement vs R1 | 78 | 1,005 |
| | fraction diverged vs R1 | 236 | 114 |

- **Perplexity segments, KL against R1, are the end-to-end evaluation.**
  - Against R0 the BF16 designs are indistinguishable, because BF16 input
    rounding swamps them.
  - Ada's truncating FP8 accumulator is visible even against R0 (46
    segments, +2.6% KL), being only ~2^3.6 below the quantization's
    error, and biased.
- **Decode ranks poorly.**
  - Under `bf16`, R0's own greedy replies are high-confidence text: both
    designs' KL against R1 is 2.0e-5.
  - Under `fp8-row`, every run diverges within ~45 tokens, so the index is
    uninformative there.
  - It stays as the generation view; forced disagreement and KL, not the
    divergence index, are its numbers.
- **Seconds per segment:** R0 0.3, R1 0.7, Ampere 10.5, CDNA3 17.3, Ada
  4.4, Blackwell 4.7.
- **Defaults set:**
  - `perplexity.py --segments 50`: 39 with a margin for the spread's
    uncertainty; 0 runs all 146.  That is 4-15 min per design.
  - `decode.py --prompts 30`.
  - `zeroshot.py --items 2000` is unchanged and unmeasured.
- **Memory:** a forced prompt's logits (~2,300 positions x 152k vocab, 1.3
  GB in FP32) came near the TITAN V's 12 GB; the allocator retried once and
  recovered.  Qwen3.5's 248k vocabulary would need ~2.3 GB.  Phase 7 adds
  logits in blocks.
- **Not done:** a 3-seed check at the chosen sizes, and zero-shot's power;
  both are cheap to add if Stage 2 needs them.

- **What:** no code; measurements recorded here and in
  `mmasim-serving.md`.
  - Qwen3-0.6B, R0, one BF16 design and one FP8 design, at several subset
    sizes, all against the full pass: 146 segments; the full suite.
  - Three seeds at the chosen size.
  - The time per design at that size.
- **Pick** sizes by power analysis (Miller 2024): from the full pass's
  per-unit variance of the paired differences, the smallest sizes that
  detect Ada vs Blackwell under `fp8-row` and Ampere vs CDNA3 under `bf16`,
  the widest local separations.  They become the defaults.
- **Why last:** it depends on every earlier phase.  It is what makes the
  "local predicts end-to-end" check affordable.

### Phase 7 -- Easy performance wins

**Done.**  The regression net holds: `local.py`'s JSON is identical for
every layer and total, one pass against one design at a time (`bf16` at 512
and 2048 tokens; `fp8-row`, `nvfp4` at 512).  `test_local.py` checks it
too, by group and against both references.

- **Seconds, `bf16`, five designs** (before: each design's total; after:
  its kernels, and the metrics once for all):

  | tokens | before | after | wall |
  |---|---|---|---|
  | 512 | 16.0 (1.6-5.5 each) | 10.1 kernels + 2.8 metrics | 43 -> 37 s |
  | 2048 | 58.1 (5.5-20.9 each) | 39.5 kernels + 10.8 metrics | 86 -> 75 s |

- **The shared FP64 GEMMs are the smaller part.**  At 2048 tokens, CDNA2's
  metrics were 3.7 s alone and five designs' are 10.8 s.  That splits into
  ~1.9 s shared and ~1.8 s per design, for the elementwise comparison and
  its host syncs.  So an extra design costs its kernel plus ~1.8 s at
  2048 tokens, or ~0.4 s at 512.  A screen that wants less restricts `-m`
  (normwise alone), or fuses the comparison.
- **Outputs held at once:** `_OUT_ELEMS` (2^28 FP32 elements) bounds the
  designs' outputs per layer, so `lm_head` at 2048 tokens runs one design
  at a time.
- **Parallel compile:** `kernels.precompile`, 13 designs in 9.2 s against
  ~38 s in series (FPy's compile is 1.4-5.4 s per design; Triton's JIT
  ~0.03 s, warm).  `kernels.compiled`'s cache became a dict so that it can
  be filled.  A `register`ed design compiles in series, since MMA-Sim's
  builds are lambdas and go to workers by name.  `local.py -j` sets the
  processes.  Python 3.14's `forkserver` re-imports the main script, so a
  caller needs a `__main__` guard.
- **Activation cache:** `local.py --acts DIR`, opt-in, since a cache
  outlives a code change.  Keyed by model, scheme, weights' source,
  workload, tokens, seed and transcripts.  MT-Bench at 512 tokens: 34 s ->
  13.5 s wall, a 207 MB file.
- **Logits in blocks:** `perplexity._log_probs` runs the decoder once and
  applies `lm_head` and the log-softmax per block of 512 positions.
  `Totals.compare` consumes the blocks, and the references are gathered on
  the host.
  - A 2,300-token sequence peaks 0.88 GiB above the model, against 2.47 GiB
    before, with the NLL identical.
  - A run is its own reference by `None`, since its blocks are not on the
    host until it is done.
  - `seconds` now includes the comparison.

Changes that are obvious and cheap, for Stage 2's sweep and Stage 3's
screen (`mmasim-serving.md`, Roadmap).  They come after Phase 6, since its
runs start a new process per step and would pick up edits made mid-run.

- **Several designs per pass in `local.evaluate`** (`designs: list[str]`):
  - Loop layers and blocks outer, designs inner.
  - The design-independent FP64 work is done once per block: the exact
    quantized product, `|x|ᵀ|w|`, and the same for the unquantized
    operands, up to four GEMMs.  At 2048 tokens that is ~10 TFLOP, most
    of a fast design's time (CDNA2: 3.7 s in all).
  - No memory beyond one block, unlike caching whole references (~8 GB
    each).
  - `metrics.local` takes the reference products instead of computing
    them.
- **Captured activations cached on disk** (`torch.save`), keyed by model,
  scheme, weights' source, workload, tokens and seed, so a sweep captures
  once across invocations.  MT-Bench is 80 prefills.
- **Logits in blocks of positions:** apply `lm_head` to the final hidden
  states a block at a time, so a long forced sequence never holds the full
  `[t, vocab]` logits.  Phase 6 came within one allocator retry of OOM on
  Qwen3-0.6B; Qwen3.5's vocabulary is 1.6x larger.
- **Designs compiled in parallel ahead of evaluation** (a process pool, as
  `compile_triton.py -j` does), so a sweep's compile time is spread over
  cores rather than paid in series on first use.
- **Regression net:** `local.py`'s JSON identical, all designs of a scheme
  in one pass against one at a time.
- **Measured:** seconds per design before and after, at 512 and 2048 tokens.
- **Tests:**

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_local.py

### Phase 8 -- Restructure `serve/`

**Done.**  The scripts import only `core`, and `core` imports no script:
the four script-to-script imports are gone (`workloads` from `perplexity`
and `decode`, `chat` from `decode`, `decode` from `perplexity`, `local`'s
`perplexity.CONTEXT`).  Where it departed:
- **Two more `core` modules than planned.**  `scoring.py` holds
  `perplexity.py`'s `Totals`, `evaluate` and `against`, which `decode.py`
  shares.  `workloads` also holds the MATH-500 prompts.
- **`cli.py`** holds `add_args`, `add_scheme_args` (was `checkpoints.add_args`)
  and `runs`.
- **Relative imports in `core`** (`from . import quant`).  `kernels` finds
  `compile.py` and `compile_triton.py` at `parents[2]`.
- **Tests** import a script only to test its own functions
  (`decode.divergences`, `local`, `zeroshot`).
- **Regression net:** the serve suite passes (80), `local.py`'s JSON at 512
  tokens is identical to Phase 7's, and every script runs on real data.

- **What:** library code moves into a package, `serve/core/`; the top level
  keeps only the scripts a user runs.

  ```
  serve/
    local.py perplexity.py zeroshot.py decode.py chat.py    user-facing
    core/
      quant.py kernels.py swap.py checkpoints.py metrics.py
      workloads.py   WikiText-2 (wikitext, segments, CONTEXT from perplexity.py),
                     MT-Bench, MATH-500 prompts, pick (from swap.py)
      generate.py    stop_tokens, encode, stream, greedy (from decode.py)
      cli.py         add_args, runs (from swap.py and checkpoints.py)
    tests/
  ```

  Scripts import `core`; `core` imports no script.
- **Why:** today `workloads.py` imports `perplexity.py` and `decode.py`,
  scripts, for their data and generation helpers.  That is why `pick` had to
  live in `swap.py` (Phase 2).  With the library under `core/`, the helpers
  sit where they belong and the cycle cannot recur.
- **Why last:** it only moves code, after every phase has settled what the
  code is.
- **Regression net:** the full serve suite, and `local.py` on 512 WikiText-2
  tokens identical before and after, as in Phase 1.
- **Tests:** the whole serve suite, since every import changes.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests

### After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n auto
cd examples/mmasim && ../../.venv/bin/python -m pytest tests serve/tests -q
.venv/bin/python -m mypy fpy2
.venv/bin/ruff check examples/mmasim/serve
```

Then update `mmasim-serving.md`: link this plan from Nice to have, and point
the "local metrics predict end-to-end effects" gap at it.

## Open items

Settled in review:

- **Zero-shot size:** one `--items` cap for every task, default 2,000.  The
  task sizes differ (PIQA 1,838; ARC-e 2,376; ARC-c 1,172; HellaSwag
  10,042; WinoGrande 1,267; LAMBADA 5,153), so the cap also subsamples ARC-e
  and LAMBADA.  R0's full-suite numbers stay as they are.  Reopen if the
  capped R0 falls outside a standard error of the full one.
- **Teacher-forced divergence replaces free running.** `greedy` stays, so a
  few designs can be run free for comparison with Yuan et al.  Reopened by
  Phase 5: the two agree for `bf16-exact` but not prompt by prompt for the
  designs.  See the open item below.
- **The divergence reference is R0's forced argmax,** not its generated
  tokens: the same computation path as the design's, so a difference is
  the design's and not prefill vs decode.  Reopen if Phase 5's floor
  measurement shows the two rarely differ.
- **Uncertainty: paired normal SEs over the unit** (segment, item,
  prompt), which is Miller 2024's recommendation for means.
  - Every statistic here is a mean but one; a PPL ratio's interval is
    `exp` of the ΔNLL interval.
  - The exception is the median divergence index, which gets a paired
    bootstrap.
  - Holm's correction applies across the designs compared.

### Do forced and free-running divergence agree in rate?

Phase 5 found they differ prompt by prompt at near-ties, from prefill's and
decoding's attention rounding.  If they agree in rate, forced stands in for
Yuan et al.'s numbers.  If not, the paper reports free running for the few
designs it quotes, and forced for the grid.

**Provisional:** forced, with its disagreement rate and KL as the headline,
since they average over positions and don't hinge on one near-tie.
**Settle** by comparing the fraction diverged on ~20 prompts for two designs:
free running costs ~12 min per design.

### Not in this plan

Uncertainty on local metrics is its own essential gap in
`mmasim-serving.md`: SEs clustered by sequence (MT-Bench positions within a
conversation).  It can reuse `pick` and Phase 4's statistics.

## Sources

- Miller, *Adding Error Bars to Evals: A Statistical Approach to Language
  Model Evaluations*, 2024 -- https://arxiv.org/abs/2411.00640
- Koehn, *Statistical Significance Tests for Machine Translation
  Evaluation*, EMNLP 2004 (the paired bootstrap)
