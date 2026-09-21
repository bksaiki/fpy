"""
Examples hosted on readthedocs.io and in the README.

Run this module to print the tables shown there:

    python3 examples/website.py
"""

import fpy2 as fp


@fp.fpy
def dot(xs: list[fp.Real], ys: list[fp.Real], K: int, block: fp.Context) -> fp.Real:
    """A blocked dot product: exact products, blocks of `K` summed in `block`."""
    assert len(xs) == len(ys) and len(xs) % K == 0
    acc = 0
    for start in range(0, len(xs), K):
        with block:
            inner_acc = 0
            for x, y in zip(xs[start:start + K], ys[start:start + K]):
                with fp.REAL:
                    p = x * y       # every product is exact ...
                inner_acc += p      # ... but block sums round to `block`
        with fp.FP32:
            acc += inner_acc        # one fp32 addition per block
    return acc


@fp.fpy(ctx=fp.REAL)
def dot_ref(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    """The same dot product, with no rounding anywhere."""
    return sum([x * y for x, y in zip(xs, ys)])


@fp.fpy
def dot_prod(a: list[fp.Real], b: list[fp.Real]) -> fp.Real:
    """
    Computes the dot product of two vectors.

    Parameters:
        a: First vector.
        b: Second vector.

    Returns:
        The dot product of the two vectors, correctly rounded under the current context.
    """
    assert len(a) == len(b)
    sum: fp.Real = 0
    with fp.REAL:
        for ai, bi in zip(a, b):
            sum += ai * bi
    return fp.round(sum)


@fp.fpy
def muller(n: int) -> fp.Real:
    """Muller's recurrence: the exact sequence converges to 5."""
    a, b = 4, 4.25
    for _ in range(n):
        a, b = b, 108 - (815 - 1500 / a) / b
    return b


def blocked_dot_table():
    """The mixed-precision dot product table from the README."""
    xs = ys = [0.1] * 4096
    exact = dot_ref(xs, ys).as_rational()   # the true value, as a Fraction

    print(f'{"block":>6} {"format":>9} {"result":>11} {"rel. error":>11}')
    for K, name, ctx in [(4096, 'float16', fp.FP16), (32, 'float16', fp.FP16),
                         (4096, 'bfloat16', fp.BF16), (32, 'bfloat16', fp.BF16),
                         (32, 'float32', fp.FP32)]:
        r = dot(xs, ys, K, ctx)
        err = abs(r.as_rational() - exact) / exact
        print(f'{K:>6} {name:>9} {float(r):>11.6f} {float(err):>11.3%}')


def muller_table():
    """Muller's recurrence under each format, from the examples page."""
    formats = {
        'float16': fp.FP16,
        'bfloat16': fp.BF16,
        'float32': fp.FP32,
        'float64': fp.FP64,
        'exact': fp.REAL,
    }

    print('  n ' + ''.join(f'{name:>10}' for name in formats))
    for n in range(0, 26, 5):
        xs = [float(muller(n, ctx=ctx)) for ctx in formats.values()]
        print(f'{n:>3} ' + ''.join(f'{x:>10.4f}' for x in xs))


if __name__ == '__main__':
    blocked_dot_table()
    print()
    muller_table()
