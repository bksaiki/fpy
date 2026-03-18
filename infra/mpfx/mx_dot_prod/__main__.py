import argparse
import gzip
import pickle
import fpy2 as fp
import random
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

@fp.fpy
def dot_prod(xscale: fp.Real, xelts: list[fp.Real], yscale: fp.Real, yelts: list[fp.Real], c: fp.Real) -> fp.Real:
    """
    Computes the dot product with uniform precision
    """
    dot_prod = xscale * yscale * sum([x * y for x, y in zip(xelts, yelts)])
    with fp.FP32:
        return dot_prod + c


MIN_E8M0_BITS = fp.MX_E8M0.encode(fp.Float.from_float(float(2 ** -127)))
MAX_E8M0_BITS = fp.MX_E8M0.encode(fp.Float.from_float(float(2 ** 127)))

def sample_block():
    # sample 33 bytes of data
    bits = random.getrandbits(8 * 33)
    # scale is the upper 8 bits, elements are the lower 32 bytes
    scale_bits = bits >> (8 * 32)
    scale = fp.MX_E8M0.decode(scale_bits)
    # elements are the lower 32 bytes
    elts = []
    for i in range(32):
        elt_bits = (bits >> (8 * i)) & 0xFF
        elt = fp.MX_E5M2.decode(elt_bits)
        elts.append(elt)
    return bits, scale, elts



    # sample a scale
    scale_bits = random.randint(MIN_E8M0_BITS, MAX_E8M0_BITS)
    scale = fp.MX_E8M0.decode(scale_bits)
    # sample 32 elements
    bits = scale_bits
    elts = []
    for i in range(32):
        elt_bits = random.randint(0, 255)
        elt = fp.MX_E5M2.decode(elt_bits)
        bits |= elt_bits << (8 * i)  # pack the element bits into the block
        elts.append(elt)
    # return the scale and elements
    return bits, scale, elts


def run_trial(seed):
    # sample input
    random.seed(seed)
    xbits, xscale, xelts = sample_block()
    ybits, yscale, yelts = sample_block()
    c = 0

    # evaluate implementations
    result_dot_prod = dot_prod(xscale, xelts, yscale, yelts, c, ctx=fp.FP32)
    result_mx_dot_prod = dot_prod(xscale, xelts, yscale, yelts, c, ctx=fp.REAL)

    # compare output (careful with NaNs!)
    if result_dot_prod.isnan or result_mx_dot_prod.isnan:
        is_mismatch = not (result_dot_prod.isnan and result_mx_dot_prod.isnan)  # mismatch if one is NaN and the other isn't
    else:
        is_mismatch = result_dot_prod != result_mx_dot_prod

    return xbits, ybits, is_mismatch


def plot_mismatches(mismatch_x, mismatch_y):
    # plot mismatches (normalized to [0, 1])
    MAX_BITS = 10 ** 79
    fig, ax = plt.subplots(figsize=(8, 8))
    if mismatch_x:
        norm_x = np.array(mismatch_x, dtype=float) / MAX_BITS
        norm_y = np.array(mismatch_y, dtype=float) / MAX_BITS
        ax.scatter(norm_x, norm_y, color="crimson", label="mismatch", alpha=0.7, s=1)

    ax.set_xlim(0, 2.96)
    ax.set_ylim(0, 2.96)
    ax.set_xlabel("x block bits (normalized)")
    ax.set_ylabel("y block bits (normalized)")
    # ax.set_title(f"Block encoding space ({len(mismatch_pts)} mismatches / {N})")
    # ax.legend()
    plt.tight_layout()
    plt.savefig("dot_prod_comparison.png", dpi=300)



parser = argparse.ArgumentParser()
parser.add_argument("num_inputs", type=int, help="number of trials")
parser.add_argument("-t", "--threads", type=int, default=1, help="number of worker threads (default: all CPUs)")
parser.add_argument("-s", "--seed", type=int, default=1, help="base random seed")
parser.add_argument("-o", "--output", type=str, default="mismatches.pkl", help="file to dump mismatches (default: mismatches.pkl)")
parser.add_argument("--replot", action="store_true", help="replot from existing mismatch file instead of running new trials")
args = parser.parse_args()

num_inputs: int = args.num_inputs
seed: int = args.seed
num_threads: int = args.threads
output: str = args.output
replot: bool = args.replot

if replot:
    with gzip.open(output, "rb") as f:
        data = pickle.load(f)
    mismatch_x = data["x"]
    mismatch_y = data["y"]
    print(len(mismatch_x), "mismatches loaded from", output)
else:
    mismatch_x = []
    mismatch_y = []
    with mp.Pool(processes=num_threads) as pool:
        for xbits, ybits, is_mismatch in pool.map(run_trial, range(seed, seed + num_inputs)):
            if is_mismatch:
                mismatch_x.append(xbits)
                mismatch_y.append(ybits)

    print(len(mismatch_x), "mismatches found out of", num_inputs)

    # dump mismatches to file
    with gzip.open(args.output, "wb") as f:
        pickle.dump({"x": mismatch_x, "y": mismatch_y, "N": num_inputs, "seed": seed}, f)
    print("mismatches saved to", args.output)

plot_mismatches(mismatch_x, mismatch_y)
