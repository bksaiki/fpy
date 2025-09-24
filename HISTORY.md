# Version History

## [next] - ???
### Features:
 - Numbers:
   - implement comparators between `Float`/`RealFloat` and Python numbers
### Fixes:
 - `RealFloat`: fix `as_rational()`
 - `Float`: fix `is_positive()` and `is_negative()`

## [0.0.11] - 2025-09-17
### Features:
 - Backend
   - compiler to C++ (experimental)
 - Language
   - contexts are first-class objects
   - context constructors are just functions
   - `fpy` and `fpy_primitive` decorators accept explicit keywords
      - all metadata goes passed as a dictionary to the `meta` keyword
   - added native support for `len`
 - Analyses
   - added reachability analysis
   - added `ContextType` to the type system
   - cleaned up context inference
 - Libraries
  - vector library (`fpy2.libraries.vector`)
  - matrix library (`fpy2.libraries.matrix`)

### Changes:
- Analyses
  - rewrote `DefineUse` analysis

### Fixes:
 - fixed evaluation of `enumerate`
 - fixed evaluation of `dim` for empty lists


## [0.0.10] - 2025-08-20
### Features:
 - Library
   - added EFT library (`fpy2.libraries.eft`)
 - Analyses
   - added type inference
   - added rounding context inference

### Fixes:
 - interpreter was ignoring stochastic rounding
 - cleanup `DefinitionUseAnalysis`

## [0.0.9] - 2025-08-08
### Features:
 - Numbers library
  - added `ExpContext` for (bounded) exponential numbers.
  - added `MX_E8M0` and `MX_INT8` context aliases
 - Language
  - add `round_at` builtin

### Changes:
 - Numbers library
   - renamed `ExtFloatContext` to `EFloatContext`

### Fixes
 - Numbers library
   - fixed `encode()` for `EFloatContext`
   - fixed `round()` for special values in `EFloatContext`
 - Language
   - parse Python `abs` corrrectly

## [0.0.8] - 2025-07-29
### Features:
 - Numbers library
   - added `SMFixedContext` for signed-magnitude fixed-point numbers.
 - Language
   - native support for Python's `sum`, `max`, and `min` functions.

### Changes:
- Runtime:
  - adds `pathos` for multiprocessing support
 - Language
   - separate `tuple` and `list` types


## [0.0.7] - 2025-07-21
### Features:
 - Numbers library
   - added overflow modes `OV` for floating-point contexts.
   - added stochasic rounding options to `round()` and `round_at()`.
   - implements `encode()` and `decode()` methods for `FixedContext`
 - Language
   - added `round_exact` primitive

### Changes:
  - Runtime;
    - performance improvements to AST nodes
  - Language
    - rename `round` to `roundint`
    - rename `cast` to `round`

## [0.0.6] - 2025-07-08
### Features:
 - Added `fpy2.libraries.base` to contain all builtins (automatically imported).

### Changes:
 - Named constants, e.g., `NAN`, changed to constant functions, e.g., `nan()`.
 - Performance improvements for AST visitors
 - Performance improvements for interpreters

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
