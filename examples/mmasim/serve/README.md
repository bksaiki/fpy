# End-to-end LLM evaluation

Runs Qwen3.5-0.8B with every linear layer's matmul computed by a compiled MMA
design, and measures how far each design moves the model's next-token
predictions from a reference. The design space is `sweep.grid()`: each
generation's BF16 and FP8 instruction (Blackwell/Hopper, Ada, CDNA3, CDNA2;
Blackwell FP8 is MXFP8), swept over the alignment width `F` (the paper's
`k`), each also with `c` added the other vendor's way.

## Setup

From the repository root:

```sh
uv sync && source .venv/bin/activate
uv pip install torch triton -r examples/mmasim/serve/requirements.txt
```

(`uv sync` removes packages installed by hand; rerun the second line after
it.) A CUDA GPU is required. Do not install `causal-conv1d` or
`flash-linear-attention`: `transformers` would then run Qwen3.5's linear
attention through their kernels, and every result would move. Everything below runs from `examples/mmasim` and
writes under `results/` (`--results` to change it). The model and WikiText-2
download from Hugging Face on first use.

## Running

```sh
cd examples/mmasim
python serve/sweep.py --manifest --model Qwen/Qwen3.5-0.8B -j 1
python serve/sweep.py --model Qwen/Qwen3.5-0.8B --segments 10 --seed 0
python serve/plot_grid.py --table sweep-grid-Qwen3.5-0.8B.csv
```

1. **`--manifest`** compiles every grid design and launches it as the
   evaluation will (at each of the model's layer widths), writing
   `results/sweep/manifest.csv`: a row per design with its parameters, name,
   status (`ok` or `refused`, with the reason) and
   `compile_seconds`. `-j 1` compiles one design at a time, so each compile
   time is its own; `-j 8` is faster. Expect 8 refusals: CDNA3 FP8 with `c`
   added with the products.
2. **`sweep.py`** runs `frozen.py` once per quantization scheme
   (`bf16`, `fp8-row`, `fp8-row:fnuz`, `mxfp8`) over 10 random 2048-token
   WikiText-2 segments. Each design runs twice: *amplified* (frozen: each
   layer gets the reference's quantized input, plus its own input's
   difference times the weights) and *observed* (free: as deployed). Each
   scheme also runs its reference with ±1 ulp of random error, the plots'
   floor. `--model` is Qwen3.5-0.8B by default, or Qwen3-0.6B.
3. **`plot_grid.py`** draws `results/plots/sweep-grid-Qwen3.5-0.8B.{png,pdf}`
   from the table alone.

The full grid (105 designs) takes about 13 h on a TITAN V: ~25 min to
compile, then 3-9 min per design, ~30 min for each CDNA2 design with `c`
added after the products. Run it detached, e.g. with the three commands
above in `run.sh`:

```sh
tmux new-session -d -s grid "bash run.sh 2>&1 | tee results/sweep/grid.log"
```

## Outputs

`results/sweep/sweep-grid-Qwen3.5-0.8B.csv`, a row per grid point:

| Column | Meaning |
|---|---|
| `anchor`, `value`, `design` | base instruction, the point's label, the design's name |
| `status`, `reason` | `ok`, or `refused` and why |
| `a` ... `e_zero` | the design's parameters (`F`, `join`, `L`, `rho`, ...) |
| `kl`, `kl_se` | amplified KL(ref ‖ design), mean and standard error over segments |
| `free_kl`, `free_kl_se` | observed KL |
| `floor_kl`, `k0_kl` | the scheme's ±1 ulp floor, amplified and observed |
| `seconds`, `free_seconds` | run time over all segments, amplified and observed |
| `compile_seconds` | time to compile the kernels the design ran |

Beside it, `sweep-grid-Qwen3.5-0.8B.provenance.json` records the command and
commit. `e2e-Qwen3.5-0.8B-<scheme>.{json,csv}` are each scheme's raw
`frozen.py` output; each run overwrites them, so after a partial rerun they
hold only that run's designs. The table is the complete record.

## Rerunning

The grid's table grows: a rerun keeps its `ok` rows whose design (named by
a digest of its parameters) is still in the grid, and runs only the rest
(new or changed points, and refused ones again). After changing the
compiler or the run settings, delete the table for a fresh run. Compiled
kernels are cached in `results/cache/kernels`, keyed by the design and the
compiler's sources; delete it to measure compile times afresh.

To replot elsewhere, the table is all `plot_grid.py` reads:

```sh
tar czf grid-data.tar.gz results/sweep/sweep-grid-Qwen3.5-0.8B.* results/sweep/manifest.csv
python serve/plot_grid.py --table sweep-grid-Qwen3.5-0.8B.csv
```
