import argparse
import fpy2 as fp
import matplotlib.pyplot as plt
import numpy as np
import random

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    inputs: list[fp.RealFloat]
    num_randbits: int | None
    k: int
    seed: int
    rm: fp.RM

@dataclass(frozen=True)
class Result:
    pts: list[tuple[float, float]]
    mean_bias: float   # mean(P(round up) - dist); 0 = unbiased
    mae: float         # mean |P(round up) - dist|


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
    return Result(
        pts=pts,
        mean_bias=float(np.mean(biases)),
        mae=float(np.mean(np.abs(biases))),
    )


if __name__ == '__main__':
    DEFAULT_NUM_INPUTS = 1000
    DEFAULT_NUM_TRIALS = 100
    R = [1, 2, 4, 8, 16, None]
    RM = fp.RM.RTO

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
    inputs: list[fp.RealFloat] = []
    while len(inputs) < num_inputs:
        t = random.random()
        if t > 0 and t < 1:
            inputs.append(fp.RealFloat.from_float(t))

    # pass plain floats to worker processes to avoid pickling RealFloat
    inputs_f = [float(x) for x in inputs]

    # measure for each `num_randbits` in R using multiprocessing
    args = [Task(inputs=inputs_f, num_randbits=num_randbits, k=num_trials, seed=seed, rm=RM) for num_randbits in R]
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(run_trial, args))

    # print bias summary
    print(f"{'num_randbits':>14}  {'mean bias':>12}  {'MAE':>12}")
    for num_randbits, result in zip(R, results):
        label = str(num_randbits) if num_randbits is not None else 'None'
        print(f"{label:>14}  {result.mean_bias:>+12.6f}  {result.mae:>12.6f}")

    # make a subplot for each `num_randbits`
    fig, axs = plt.subplots(1, len(R), figsize=(5 * len(R), 5))
    fig.suptitle(f'Stochastic Rounding Bias (RM={RM.name}, N={num_inputs}, k={num_trials})')

    for i, (num_randbits, result) in enumerate(zip(R, results)):
        axs[i].scatter(*zip(*result.pts), s=4)
        axs[i].plot([0, 1], [0, 1], color='red', linewidth=0.8, linestyle='--', label='ideal')
        axs[i].set_title(f'num_randbits={num_randbits}\nbias={result.mean_bias:+.4f}, MAE={result.mae:.4f}')
        axs[i].set_xlabel('normalized distance (ULP)')
        axs[i].set_ylabel('P(round up)')

    plt.tight_layout()
    plt.savefig(f'sweep_{RM.name}_N{num_inputs}_k{num_trials}.png', dpi=300)
