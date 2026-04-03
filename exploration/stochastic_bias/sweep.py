import argparse
import fpy2 as fp
import matplotlib.pyplot as plt
import numpy as np
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    idx: int
    inputs: list[fp.RealFloat]
    num_randbits: int | None
    k: int
    seed: int
    rm: fp.RM

@dataclass(frozen=True)
class Result:
    pts: list[tuple[float, float]]
    mean_bias: float     # mean(P(round up) - dist); 0 = unbiased
    mean_avg_err: float  # mean |P(round up) - dist|
    variance: float      # variance of (P(round up) - dist)


def run_trial(task: Task):
    rng = np.random.default_rng(task.seed)
    pts: list[tuple[float, float]] = []
    for x in task.inputs:
        x = fp.RealFloat.from_float(x)
        rtz = x.round(min_n=-1, rm=fp.RM.RTZ)
        raz = x.round(min_n=-1, rm=fp.RM.RAZ)

        dist = (float(x) - float(rtz)) / (float(raz) - float(rtz))

        num_round_up = 0
        for _ in range(task.k):
            # round `x` stochastically to `num_randbits` bits
            y = x.round(min_n=-1, rm=task.rm, num_randbits=task.num_randbits, rng=rng)
            if y == raz:
                num_round_up += 1

        # record the (distance, emperical probability of rounding up) pair
        pts.append((float(dist), num_round_up / task.k))

    biases = [p - d for d, p in pts]
    mean_bias = float(np.mean(biases))
    mean_avg_err = float(np.mean(np.abs(biases)))
    variance = float(np.var(biases))

    return Result(
        pts=pts,
        mean_bias=mean_bias,
        mean_avg_err=mean_avg_err,
        variance=variance,
    )


def print_comparison_table(
    rm1: fp.RM,
    rm2: fp.RM,
    R: list[int | None],
    results1: list[Result],
    results2: list[Result],
) -> None:
    """Print a grid comparing (R1, rm1) vs (R2, rm2) for bias diff and variance diff."""
    r_labels = [str(r) if r is not None else 'None' for r in R]
    col_w = 22
    label_w = 6

    header = f"  {'R1\\R2':>{label_w}}  " + "  ".join(f"{lbl:^{col_w}}" for lbl in r_labels)
    sep = "-" * len(header)
    print(f"\n=== {rm1.name} vs {rm2.name} ===")
    print(f"  (Δ|B| = |bias[R1,{rm1.name}]| - |bias[R2,{rm2.name}]|,  ΔV = var[R1,{rm1.name}] - var[R2,{rm2.name}])")
    print(sep)
    print(header)
    print(sep)
    for r1_lbl, res1 in zip(r_labels, results1):
        row = f"  {r1_lbl:>{label_w}}  "
        cells = []
        for res2 in results2:
            d_bias = abs(res1.mean_bias) - abs(res2.mean_bias)
            d_var  = res1.variance  - res2.variance
            cell = f"Δ|B|={d_bias:+.4f} ΔV={d_var:+.4f}"
            cells.append(f"{cell:^{col_w}}")
        row += "  ".join(cells)
        print(row)
    print(sep)


if __name__ == '__main__':
    DEFAULT_NUM_INPUTS = 1000
    DEFAULT_NUM_TRIALS = 100
    R = [1, 2, 4, 8, 16, None]
    RMS = [fp.RM.RNE, fp.RM.RTZ, fp.RM.RTO]

    parser = argparse.ArgumentParser(description='Sweep stochastic rounding bias')
    parser.add_argument('-n', type=int, default=DEFAULT_NUM_INPUTS, help='number of input samples (default: 1000)')
    parser.add_argument('-k', type=int, default=DEFAULT_NUM_TRIALS, help='number of rounding trials per sample (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--threads', type=int, default=1, help='number of worker processes (default: CPU count)')
    args = parser.parse_args()

    num_inputs: int = args.n
    num_trials: int = args.k
    seed: int = args.seed
    num_threads: int = args.threads

    random.seed(seed)

    # sample `N` inputs on U(0, 1)
    inputs: list[float] = []
    while len(inputs) < num_inputs:
        t = random.random()
        if t > 0 and t < 1:
            inputs.append(t)

    # run trials for every (rm, num_randbits) combination
    tasks: list[Task] = []
    for rm in RMS:
        for num_randbits in R:
            idx = len(tasks)
            tasks.append(Task(
                idx=idx,
                inputs=inputs,
                num_randbits=num_randbits,
                k=num_trials,
                seed=seed,
                rm=rm,
            ))

    print(f"Running {len(tasks)} tasks with {num_threads} threads...")
    all_results: list[Result | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(run_trial, task): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            all_results[task.idx] = fut.result()
            print(f"finished [{task.idx + 1}/{len(tasks)}, rm={task.rm.name}, R={task.num_randbits}]")

    results: list[Result] = []
    for res in all_results:
        assert res is not None
        results.append(res)

    # reshape: results_by_rm[rm] = list of Result (one per R value)
    results_by_rm: dict[fp.RM, list[Result]] = {}
    for idx, rm in zip(range(0, len(tasks), len(R)), RMS):
        results_by_rm[rm] = results[idx : idx + len(R)]

    # per-RM bias summary
    for rm in RMS:
        print(f"\n--- {rm.name} ---")
        print(f"  {'num_randbits':>14}  {'mean bias':>12}  {'MAE':>12}  {'variance':>12}")
        for num_randbits, result in zip(R, results_by_rm[rm]):
            label = str(num_randbits) if num_randbits is not None else 'None'
            print(f"  {label:>14}  {result.mean_bias:>+12.6f}  {result.mean_avg_err:>12.6f}  {result.variance:>12.6f}")

    # pairwise comparison tables
    for i, rm1 in enumerate(RMS):
        for j, rm2 in enumerate(RMS):
            if j <= i:
                continue
            print_comparison_table(rm1, rm2, R, results_by_rm[rm1], results_by_rm[rm2])

    # plots for each RM (one figure per RM)
    for rm in RMS:
        fig, axs = plt.subplots(1, len(R), figsize=(5 * len(R), 5))
        fig.suptitle(f'Stochastic Rounding Bias (RM={rm.name}, N={num_inputs}, k={num_trials})')
        for i, (num_randbits, result) in enumerate(zip(R, results_by_rm[rm])):
            axs[i].scatter(*zip(*result.pts), s=4)
            axs[i].plot([0, 1], [0, 1], color='red', linewidth=0.8, linestyle='--', label='ideal')
            axs[i].set_title(f'num_randbits={num_randbits}\nbias={result.mean_bias:+.4f}, MAE={result.mean_avg_err:.4f}')
            axs[i].set_xlabel('normalized distance (ULP)')
            axs[i].set_ylabel('P(round up)')

        plt.tight_layout()
        plt.savefig(f'sweep_{rm.name}_N{num_inputs}_k{num_trials}.png', dpi=150)
