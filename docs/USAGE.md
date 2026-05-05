# FPy | Usage Guide

Important links:
 - PyPI package: [fpy2](https://pypi.org/project/fpy2/)
 - Documentation: [fpy.readthedocs.io](https://fpy.readthedocs.io/)
 - GitHub: [fpy](https://github.com/bksaiki/fpy)
 - Guide: [USAGE.md](docs/USAGE.md)

FPy is an embedded Python domain-specific language (DSL) for specifying and simulating numerical algorithms.
It consists of two main components:
- a language with Python-like syntax for writing algorithms, and
- a runtime to execute FPy code and simulate number systems

The usage guide focuses on the first component, the FPy language.

## Principles of FPy

FPy follows a few core principles:

1. **Every number is a real number**. A number is not an integer or a floating-point number;
it's just a real number.
2. **Every operation is correctly-rounded**. When you perform an operation,
the result is the exact mathematical result followed by a rounding.
3. **Rounding is determined by rounding contexts**. Any detail of rounding—number format, rounding mode, overflow behavior, etc.—is specified by a _rounding context_.
Every operation is performed with respect to a rounding context, and the result is rounded according to that context.

The language wraps these principles in a simple, Python-like syntax.
Python code can often be directly used in FPy with minimal changes.

## Writing FPy code

To write an FPy function, use the `@fpy` decorator.

```python
import fpy2 as fp

@fp.fpy
def dot_prod(x, y):
    """Compute the dot product of two vectors."""
    res = 0
    for xi, yi in zip(x, y):
        res += xi * yi
    return res
```
The decorator tells the FPy runtime to execute the function according to the semantics of the FPy language.
The FPy language is a subset of Python with additional restrictions to ensure well-formedness.

In general, you can use variables, numbers, loops, conditions, and function calls in FPy code, just like in Python.
The valid data types are booleans, numbers, tuples, and (fixed-size) lists.

### Constants

The simplest language constructs are constants.
There are boolean constants `True` and `False`, and numeric constants like `1` and `3.14`.
Crucially, **numerical constants are interpreted as-is**, that is, they are interpreted as rational numbers.

```
@fp.fpy
def example_consts():
    t1 = True
    t2 = False
    t3 = 1
    t4 = 3.14
    t5 = fp.rational(1, 3) # 1/3 as a rational number
    t6 = fp.digits(1, 3, 2) # 1 * 2^3 as a rational number
    ...
```

### Rounding Contexts

Rounding contexts are key to controlling rounding behavior in FPy.
They are created via specific classes.

```python
@fp.fpy
def example_contexts():
    FP64 = fp.IEEE754Context(11, 64) # IEEE 754 double precision (also `fp.FP64`)
    FP32 = fp.IEEE754Context(8, 32) # IEEE 754 single precision (also `fp.FP32`)
    S8 = fp.FixedContext(True, 0, 8) # 8-bit signed integer (also `fp.SINT8`)
    ...
```

During the execution of an FPy function, there is always a _current_ rounding context which dictates the rounding behavior of all operations.
The current rounding context may be changed by using the `with` statement.

```python
@fp.fpy
def example_with():
    FP64 = fp.IEEE754Context(11, 64)
    FP32 = fp.IEEE754Context(8, 32)
    with FP64:
        # operations here are rounded according to FP64
        x = 1 / 3 # x is rounded to the nearest representable FP64 number
    with FP32:
        # operations here are rounded according to FP32
        y = 1 / 3 # y is rounded to the nearest representable FP32 number
    ...
```

Two important notes about the current rounding context:

- **if an FPy function calls another FPy function, the callee uses the current rounding context**.
This means that if you set a rounding context in one function, it will be used by any function it calls,
unless that function explicitly changes the rounding context.
- **if an FPy function is called from Python, the current rounding context is the default context,
which is an IEEE 754 double-precision floating-point context**.
This means that if you call an FPy function from Python without setting a rounding context, it will use
similar semantics to Python's built-in floating-point arithmetic.

### Operations

FPy supports a variety of operations, including arithmetic, algebraic, and transcendental functions.
**All operations are correctly-rounded according to the current rounding context**.

```python
@fp.fpy
def example_ops():
    # arithmetic
    a1 = x + y
    a2 = x - y
    a3 = x * y
    a4 = x / y
    # algebraic
    b1 = fp.sqrt(x)
    b2 = fp.cbrt(x)
    b3 = fp.pow(x, y)
    # transcendental
    c1 = fp.exp(x)
    c2 = fp.log(x)
    c3 = fp.sin(x)
    c4 = fp.cos(x)
    ...
```

### Tuples

FPy supports tuples, which are immutable sequences of values.
Unlike in Python, tuple elements can only be accessed via pattern matching, not via indexing.

```python
@fp.fpy
def example_tuples():
    t = (x, y, z) # create a tuple
    # access tuple elements via pattern matching
    a, b, c = t
    ...
```

### Lists

FPy supports fixed-size lists, which are mutable sequences of values.
List elements can be accessed via indexing, and lists can be modified via assignment.
The size of a list is determined at creation and cannot be changed.

```python
@fp.fpy
def example_lists():
    lst = [1, 2, 3, 4, 5]
    x = lst[0] # access first element
    lst[1] = 10 # modify second element
    ...
```

FPy also supports list comprehensions, which are a concise way to create lists.

```python
@fp.fpy
def example_comprehensions():
    lst = [1, 2, 3, 4, 5]
    lst2 = [x * x for x in lst] # creates [1, 4, 9, 16, 25]
    ...
```

FPy supports built-in functions like `len` and `range` that operate on lists.

```python
@fp.fpy
def example_builtin():
    lst = [1, 2, 3, 4, 5]
    n = len(lst) # n is 5
    for i in range(n):
        lst[i] = lst[i] * 2 # lst is now [2, 4, 6, 8, 10]
    for i, j in enumerate(lst):
        lst[i] = j + 1 # lst is now [3, 5, 7, 9, 11]
    res = 0
    for i, j in zip(lst, lst2):
        res += i * j # res is the dot product of lst and lst2
    ...
```

#### Slicing

FPy supports list slicing `xs[start:stop]`, but its semantics **differ
from Python's**.  Where Python silently *clamps* out-of-range bounds
(`xs[10:]` on a length-3 list returns `[]`), FPy treats out-of-range
bounds as an error: a slice expression must extract a block of *exactly*
`stop - start` elements.

The runtime check is `0 <= start <= stop <= len(xs)`; any violation
raises `IndexError` (and a non-integer bound raises `TypeError`).
Omitted bounds default to `0` and `len(xs)` respectively.

```python
@fp.fpy
def example_slicing(xs: list[fp.Real]):
    a = xs[1:3]   # OK iff len(xs) >= 3 — extracts exactly 2 elements
    b = xs[:3]    # OK iff len(xs) >= 3 — equivalent to xs[0:3]
    c = xs[2:]    # OK iff len(xs) >= 2 — equivalent to xs[2:len(xs)]
    d = xs[:]     # always OK — full copy
    # The next three would all raise IndexError at runtime:
    # e = xs[10:]    # start past end of list
    # f = xs[:100]   # stop past end of list
    # g = xs[2:1]    # start > stop
    ...
```

The motivation is to make static reasoning about slice sizes precise:
under FPy's rule, `xs[a:b]` always has size `b - a` when both bounds are
concrete (regardless of `len(xs)`), and program analyses can rely on
this without the conditional branching that Python's clamping would
require.  Programs that need Python-style "best-effort" truncation must
clamp the bounds explicitly before slicing.

## Control Flow

FPy supports control flow constructs like `if` statements and `for` loops.

### If Statements

FPy's `if` statements differ from Python's `if` statements:

- one branch `if` statements: any identifier introduced in the `if` block is not accessible outside the block
- two branch `if` statements: any identifier introduced in both branches is accessible outside the block

```python
@fp.fpy
def example_control_flow():
    if x > 0:
        y = x * 2
    # y is not accessible here because it was only introduced in the `if` block
    if x < 0:
        y = x * 2
        z = y + 1
    else:
        z = 0
    # z is accessible here, but y is not
    ...
```

### For Loops

FPy's `for` loops differ from Python's `for` loops similarly.
Any identifier introduced in the `for` block is not accessible outside the block.

```python
@fp.fpy
def example_for_loop(lst):
    acc = 0
    for x in lst:
        acc += x
    # x is not accessible here
    ...
```

### While Loops

FPy's `while` loops differ from Python's `while` loops similarly.
Any identifier introduced in the `while` block is not accessible outside the block.

```python
@fp.fpy
def example_while_loop(x):
    acc = 0
    while x > 0:
        acc += x
        x -= 1
    # x is not accessible here
    ...
```
