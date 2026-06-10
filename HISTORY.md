# Version History

## [0.1.8] - 2026-06-10
### Features:
  - Language:
    - support Python `min/max` over lists
  - Testing:
    - added program generators for PBT tests over FPy programs

## [0.1.7] - 2026-05-26
### Features:
  - AST:
    - remove `RoundExact` AST node
  - Analyses:
    - add call graph analysis
    - add `Module` abstraction

### Fixes:
  - Analyses:
    - `FormatAnalysis`
      - fix addition/multiplication with zero formats
      - fix widening
    - `TypeInfer`
      - fix type checking when unpacking tuples
  - Backend:
    - C++ compiler: fix widening types
  - Number:
    - `RealFloat`: fix `__add__` with special values
    - `MPFloatContext`, `MPSFloatContext`, and `MPBFloatContext`: fix `round()` for -0

### Removed:
  - MPFX prototype compiler
  - context inference pass

## [0.1.6] - 2026-05-26
### Features:
  - AST:
    - remove `ListSet` AST node
  - Language:
    - support multiple return statements
  - Transforms:
    - add zip elimination
    - add rounding elimination

## [0.1.5] - 2026-05-13
### Features:
  - Analyses
    - added context-use analysis
    - added format inference
  - Backend:
    - new C++ compiler
  - Number:
    - added `Format` type represent number formats
  - Misc:
    - added usage guide
    - added AGENTS.md / CLAUDE.md for agents

### Changes:
  - Analyses:
    - `IndexedAssign` treated as a functional update by default
    - cleanup array size inference

### Fixes:
  - Number:
    - flipped stochastic rounding direction
  - Language:
    - slices must have correct bounds

## [0.1.4] - 2026-04-06
### Features:
 - Numbers:
   - rounding contexts take an optional RNG for stochastic rounding

### Fixes:
  - Numbers:
    - fix stochastic rounding: number of random bits was incorrect for small values

## [0.1.3] - 2026-03-30
### Features:
 - Numbers:
   - add next float methods to `Float` and `RealFloat`
   - add overflow flag to `RealFloat`
- Runtime:
   - interpreter compiles to Python bytecode for improved performance

## [0.1.2] - 2026-03-23
### Features:
 - Language:
   - add `cast` builtin for format conversion without rounding
   - add `logb` to builtins
   - add `declcontext` to builtins
 - Analyses
   - array size inference
   - partial evaluation
 - Numbers:
   - added `Engine` interface to encapsulate arithmetic engines
 - Backend:
   - added compiler to the MPFX library (experimental)

### Changes:
 - Language:
   - make `empty` a variary operator

## [0.1.1] - 2025-11-07
### Features:
 - Runtime
   - add `Runner` instance to manage design-space exploration tasks

### Fixes:
 - Runtime:
   - interpreter now prefers returning `Float` values over `Fraction`

## [**0.1.0**] - 2025-11-04
### Changes:
 - updates benchmarks, moved under `examples/`

## [0.0.13] - 2025-10-31
### Features:
 - Language:
   - support 2-/3-argument `range`
 - Library:
   - add metrics under `fpy2.libraries.metrics`
 - Analyses
   - Add monomorphizing pass
   - Add `simplify` strategy
   - Add `split` strategy
 - Numbers:
   - add comparator/arithmetic methods for `Float`
   - add additional `Float` methods
 - Testing:
   - rounding context generator

### Fixes:
 - Parser:
   - implements modified loader to stash source

## [0.0.12] - 2025-10-02
### Features:
 - Language:
   - context statements can bind the context as a variable
   - add `pass` statement
   - arithmetic works on fractions (for some operations)
 - Analyses:
   - dead code elimination
   - constant propagation
   - constant folding
   - function inlining
 - Numbers:
   - implement comparators between `Float`/`RealFloat` and Python numbers

### Changes:
 - Language:
   - [**Semantics**] constants are unrounded

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
