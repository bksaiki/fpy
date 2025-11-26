# FPy Numbers C++ Implementation

A restricted C++ implementation of FPy's number library for high-performance floating-point operations.

## Building

### Prerequisites

- CMake 3.15 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- (Optional) GMP/MPFR for arbitrary precision support

### Build Instructions

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Build Options

- `BUILD_SHARED_LIBS`: Build shared libraries (default: ON)
- `BUILD_TESTS`: Build test executables (default: ON)
- `BUILD_EXAMPLES`: Build example programs (default: OFF)

### Build Types

- Debug: `cmake -DCMAKE_BUILD_TYPE=Debug ..`
- Release: `cmake -DCMAKE_BUILD_TYPE=Release ..`

## Running Tests

```bash
cd build
ctest
# or
./tests/run_tests
```

## Project Structure

```
fpy2/cpp/
├── CMakeLists.txt          # Main CMake configuration
├── include/fpy/            # Public headers
│   ├── context.hpp         # Floating-point contexts
│   ├── float.hpp           # Float number class
│   ├── rounding.hpp        # Rounding modes
│   └── types.hpp           # Basic type definitions
├── src/                    # Implementation files
│   ├── context.cpp
│   ├── float.cpp
│   └── rounding.cpp
└── tests/                  # Test files
    ├── CMakeLists.txt
    ├── test_context.cpp
    ├── test_float.cpp
    └── test_rounding.cpp
```

## Implementation Notes

### TODO: Core Components

1. **Context System**
   - [ ] Implement IEEEContext for standard IEEE 754 formats
   - [ ] Implement FixedContext for fixed-point arithmetic
   - [ ] Add common format constants (FP16, FP32, FP64, etc.)

2. **Float Operations**
   - [ ] Define internal representation (choose between bit-packed, components, or arbitrary precision)
   - [ ] Implement basic arithmetic (add, sub, mul, div)
   - [ ] Implement comparison operators
   - [ ] Implement special value handling (inf, nan, zero)
   - [ ] Implement conversion functions

3. **Rounding**
   - [ ] Implement rounding mode logic
   - [ ] Implement tie-breaking for NEAREST modes
   - [ ] Implement overflow handling

4. **Testing**
   - [ ] Choose and integrate testing framework (Google Test recommended)
   - [ ] Add comprehensive unit tests
   - [ ] Add edge case and corner case tests

### TODO: Optional Enhancements

- [ ] Add support for additional formats (bfloat16, tensorfloat32, etc.)
- [ ] Integrate GMP/MPFR for arbitrary precision
- [ ] Add mathematical functions (sqrt, exp, log, trig)
- [ ] Add SIMD optimizations
- [ ] Add Python bindings (pybind11)
- [ ] Generate documentation (Doxygen)

## Design Decisions to Make

1. **Internal Representation**: Choose how to store float values
   - Option A: Bit-packed (uint64_t for FP64, uint32_t for FP32)
   - Option B: Separate components (sign, exponent, mantissa)
   - Option C: Arbitrary precision library (GMP/MPFR)

2. **Context Handling**: How contexts are associated with Float values
   - Option A: Store context reference in each Float
   - Option B: Pass context to operations explicitly
   - Option C: Use thread-local default context

3. **Memory Management**: How to handle resource allocation
   - Option A: Value semantics (stack allocation)
   - Option B: Shared pointers for large values
   - Option C: Custom memory pool

## License

NOTE: Add license information to match the main FPy project.
