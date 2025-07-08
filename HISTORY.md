# Version History

## [next] - ?
### Features:
 - Added `fpy2.libraries.base` to contain all builtins (automatically imported).

### Changes:
 - Named constants, e.g., `NAN`, changed to constant functions, e.g., `nan()`.

### Removed
 - `fpy2.PythonInterpreter` has been removed. Use `fpy2.DefaultInterpreter` instead.
 - `NDArray` has been removed from runtime. Use `list` or `tuple` instead.

### Fixes
 - Runtime
   - interpreted arguments are not rounded under the body rounding context
 - Standard library
   - `libraries.core.max_e` always computes under an integer context
 - Numbers library
   - fix `round()` and `round_at()` for integer arguments
   - fix `from_float()` for NaN and infinities

## [0.0.5] - 2025-07-02
### Features
 - Frontend
   - support for `enumerate`
   - remove support for `shape`
   - support Python-style slicing
   - function identifiers are resolved to the objects they map to in Python
   - FPy builtins are objects
   - cannot call arbitrary Python functions
   - renamed `@fpy_prim` to `@fpy_primitive`

### Fixes
 - Numbers library
   - removed `__round__` (and friends) from `RealFloat
   - fixed `round()` for `MPFloatContext` and `MPSFloatContext`

## [0.0.4] - 2025-06-12
### Features
 - Numbers library
    - number types
      - `RealContext` - a context for real numbers
      - `MPFixedContext` - a fixed-point context with unbounded precision
      - `MPBFixedContext` - a bounded fixed-point context
      - `MPFixed` - the usual finite-width, fixed-point number
      - renamed floating-point contexts to have `Float` in their names
 - Frontend
   - support for `zip`
   - tuple unpacking in `for` loops
   - context in `@fpy` decorator
   - added `@fpy_prim` decorator to specify Python-backed primitives
 - Middleend
   - removed the IR
   - add copy-propagation pass
   - add context-inlining pass
 - Standard library
   - small set of primitives under `fpy2.libraries.core`
 - Infrastructure
   - fuzzing tests against FPCore (Titanic)

## [0.0.3] - 2025-05-22
### Features
 - Numbers library
   - number types
     - `RealFloat` - an arbitrary-precision floating point number encoding only real numbers
     - `Float` - an arbitrary-precision floating point number including infinities and NaN
   - abstract rounding contexts
     - `Context` - base class of all rounding contexts
     - `OrdinalContext` - base class of rounding contexts whose values are countable
     - `SizedContext` - base class of rounding contexts whose values are finite in number
     - `EncodableContext` - base class of rounding contexts whose values can be converted to bitstrings
   - concrete rounding contexts
     - `MPContext` - floating-point with finite precision but unbounded exponent
     - `MPSContext` - floating-point with finite precision and a minimum exponent
     - `MPBContext` - floating-point with finite precision, minimum exponent, and maximum value
     - `IEEEContext` - IEEE 754 floating-point
     - `ExtContext` - Extended-parameter floating-point
 - Frontend
   - explicit context syntax
   - legacy FPCore-like context syntax
   - foreign values
   - pattern matching and rewrite system

## [0.0.2] - 2025-03-20
### Features
 - Parsing via `@fpy` decorator or conversion from FPCore
 - FPy AST and IR
 - Analysis:
   - Syntax checking
 - Compiler passes:
   - SSA conversion
   - verify pass
   - `if` statement conversion
   - `for` and `while` loop bundling
 - Backend:
   - FPCore
   - FPy
 - Evaluators:
   - native Python
   - Titanic
   - Real (Rival)

## [0.0.1] - 2025-01-07
### Package release
 - Initial release of the package.
