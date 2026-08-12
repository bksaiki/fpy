"""
Shared helpers for the differential sweeps in `validate_nv.py` and
`validate_amd.py`: the five input distributions and the bit-exact
output comparison.

Importing this module requires `torch` (see requirements.txt); the
validators import it lazily so their directed tiers run with only
fpy2 installed.
"""

import math
import random

import torch  # type: ignore[import-not-found]

import fpy2 as fp
import utils

# formats with no infinity encoding
NO_INF = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)

def rand_bits(n, dtype):
    """n random bit-patterns of the given dtype."""
    nbits = torch.finfo(dtype).bits
    if nbits == 16:
        raw = torch.randint(0, 2**16, (n,), dtype=torch.int32).to(torch.uint16)
    elif nbits == 32:
        return torch.randint(-2**31, 2**31, (n,), dtype=torch.int32).view(dtype)
    else:
        raw = torch.randint(0, 256, (n,), dtype=torch.uint8)
    return raw.view(dtype)

def rand_edge(n, dtype):
    """each element drawn from the format's edge values."""
    fi = torch.finfo(dtype)
    vals = [0.0, -0.0, fi.smallest_normal * fi.eps, -fi.smallest_normal * fi.eps,
            fi.smallest_normal, -fi.smallest_normal, fi.tiny, 1.0, -1.0,
            fi.max, -fi.max, fi.smallest_normal * 4.0]
    if dtype not in NO_INF:
        vals += [float('inf'), float('-inf')]
    vals += [float('nan')]
    return torch.tensor([random.choice(vals) for _ in range(n)]).to(dtype)

def rand_mixed(n, dtype):
    """random bits, but ~1/3 of entries forced to +/-0 (stresses e_zero)."""
    t = rand_bits(n, dtype)
    for i in range(n):
        r = random.random()
        if r < 0.25:
            t[i] = 0.0
        elif r < 0.35:
            t[i] = -0.0
    return t

def rand_normal(n, dtype):
    """standard normal, rounded into the input format."""
    return torch.randn(n).to(dtype)

def rand_subnormal(n, dtype):
    """normal scaled so that 1 becomes 2^emin (subnormal-centered)."""
    emin = int(torch.log2(torch.tensor(torch.finfo(dtype).smallest_normal,
                                       dtype=torch.float64)))
    return (torch.randn(n, dtype=torch.float64) * 2.0 ** emin).float().to(dtype)

GENS = [rand_bits, rand_edge, rand_mixed, rand_normal, rand_subnormal]

def out_bits(t):
    if t.dtype == torch.float16:
        return t.view(torch.int16).item() & 0xFFFF
    return t.view(torch.int32).item() & 0xFFFFFFFF

def same(ref_t, fpy_x, out_dtype=torch.float32):
    """Bit-exact comparison of a reference tensor and an FPy result."""
    if torch.isnan(ref_t) and fp.isnan(fpy_x):
        return True
    mine = torch.tensor(float(fpy_x), dtype=out_dtype)
    return out_bits(ref_t) == out_bits(mine)

def tf32_list(t):
    """FP32 tensor -> list of TF32-truncated floats (NaN passes through)."""
    return [x if math.isnan(x) else float(utils.RZ_TF32.round(x))
            for x in t.float().tolist()]
