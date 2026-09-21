# FPy

An embedded Python DSL for specifying and simulating numerical algorithms.

Important links:
 - PyPI package: [fpy2](https://pypi.org/project/fpy2/)
 - Documentation: [fpy.readthedocs.io](https://fpy.readthedocs.io/)
 - GitHub: [fpy](https://github.com/bksaiki/fpy)
 - Guide: [USAGE.md](docs/USAGE.md)

## Example

FPy is Python with explicit control of both the mathematics and
rounding: every operation is correctly rounded by the *rounding context*
it appears in, and contexts are ordinary values you can pass around —
including `fp.REAL`, which never rounds.  Here is a blocked dot product:
the products are exact, each block of `K` elements is summed in a narrow
format, and the running total is `float32`.

```python
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

xs = ys = [0.1] * 4096
exact = dot_ref(xs, ys).as_rational()   # the true value, as a Fraction

print(f'{"block":>6} {"format":>9} {"result":>11} {"rel. error":>11}')
for K, name, ctx in [(4096, 'float16', fp.FP16), (32, 'float16', fp.FP16),
                     (4096, 'bfloat16', fp.BF16), (32, 'bfloat16', fp.BF16),
                     (32, 'float32', fp.FP32)]:
    r = dot(xs, ys, K, ctx)
    err = abs(r.as_rational() - exact) / exact
    print(f'{K:>6} {name:>9} {float(r):>11.6f} {float(err):>11.3%}')
```

which prints

```
 block    format      result  rel. error
  4096   float16   32.000000     21.875%
    32   float16   40.968750      0.021%
  4096  bfloat16    4.000000     90.234%
    32  bfloat16   40.250000      1.733%
    32   float32   40.959972      0.000%
```

A flat `float16` accumulator stalls out once the running total dwarfs
the product being added to it; summing 32 elements at a time keeps the
error three orders of magnitude smaller.

See the [usage guide](docs/USAGE.md) for the language and the
[examples page](https://fpy.readthedocs.io/en/latest/example.html) for
more.

## Installation

FPy can be installed from PyPI with either `uv` or `pip`, or built from
source for development.  The following instructions assume a `bash`-like
shell.

### Installing from PyPI

Requirements:
 - Python 3.11 or later

To install the latest stable release of FPy, run:
```bash
uv pip install fpy2
```
or, with `pip`:
```bash
pip install fpy2
```

### Installing from source

Requirements:
 - Python 3.11 or later
 - `make`

#### With `uv` (preferred)

[uv](https://docs.astral.sh/uv/) is the recommended development
workflow — it handles the virtual environment and dependency
installation in a single step:
```bash
uv sync
```
This creates `.venv/` and installs FPy in editable mode along with the
`dev` dependency group.  Activate the environment with
```bash
source .venv/bin/activate
```
or prefix individual commands with `uv run` (e.g. `uv run pytest tests/unit`).

#### With `pip` (legacy)

This path is preserved for compatibility with existing tooling; new
contributors should prefer `uv` above.

If you do not have a Python virtual environment,
create one using
```bash
python3 -m venv .venv/
```
and activate it using
```bash
source .venv/bin/activate
```
To install an instance of FPy for development, run:
```bash
pip install -e .[dev]
```

To uninstall FPy, run:
```bash
pip uninstall fpy2
```

### Testing

There are a number of tests that can be run through
the `Makefile` including
```bash
make lint
```
to ensure formatting and type safety;
```bash
make unittest
```
to run the unit tests;
```bash
make infratest
```
to run the infrastructure tests.
