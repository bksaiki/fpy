# Version History

## [0.0.3] - ???
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
