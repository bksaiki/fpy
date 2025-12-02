# FPy Numbers C++ Implementation

A C++ implementation of FPy's number library
for high-performance software floating-point operations.

## Building

### Prerequisites

- CMake 3.15 or higher
- C++20 compatible compiler
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
