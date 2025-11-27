#include <bit>
#include <cfenv>
#include <cmath>

#include "fpy/engine.hpp"

namespace fpy {

namespace engine {


/// @brief Clears all floating-point exceptions.
inline void clear_exceptions() {
    std::feclearexcept(FE_ALL_EXCEPT);
}

/// @brief Sets the current rounding mode.
inline void set_rm(int rm) {
    std::fesetround(rm);
}

/// @brief Sets the rounding mode to round-to-zero (RTZ).
/// Returns the old rounding mode.
inline int set_rtz() {
    const int old_rm = std::fegetround();
    set_rm(FE_TOWARDZERO);
    return old_rm;
}

/// @brief Loads the current floating-point exceptions.
inline fexcept_t load_exceptions() {
    fexcept_t fexps;
    std::fegetexceptflag(&fexps, FE_ALL_EXCEPT);
    return fexps;
}

double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "rto_add: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    clear_exceptions();
    const int rm = set_rtz();

    // perform addition in RTZ mode
    double result = x + y;

    // load exceptions
    const auto fexps = load_exceptions();

    // reset rounding mode
    set_rm(rm);

    // check if overflow or underflow occurred
    FPY_ASSERT(
        !(fexps & (FE_OVERFLOW | FE_UNDERFLOW)),
        "rto_add: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & FE_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}

double sub(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "sub: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    clear_exceptions();
    const int rm = set_rtz();

    // perform subtraction in RTZ mode
    double result = x - y;

    // load exceptions
    const auto fexps = load_exceptions();

    // reset rounding mode
    set_rm(rm);

    // check if overflow or underflow occurred
    FPY_ASSERT(
        !(fexps & (FE_OVERFLOW | FE_UNDERFLOW)),
        "sub: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & FE_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}

double mul(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "mul: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    clear_exceptions();
    const int rm = set_rtz();

    // perform multiplication in RTZ mode
    double result = x * y;

    // load exceptions
    const auto fexps = load_exceptions();

    // reset rounding mode
    set_rm(rm);

    // check if overflow or underflow occurred
    FPY_ASSERT(
        !(fexps & (FE_OVERFLOW | FE_UNDERFLOW)),
        "mul: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & FE_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}

double div(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "div: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    clear_exceptions();
    const int rm = set_rtz();

    // perform division in RTZ mode
    double result = x / y;

    // load exceptions
    const auto fexps = load_exceptions();

    // reset rounding mode
    set_rm(rm);

    // check if overflow or underflow occurred
    FPY_ASSERT(
        !(fexps & (FE_OVERFLOW | FE_UNDERFLOW)),
        "div: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & FE_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}

double sqrt(double x, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "sqrt: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    clear_exceptions();
    const int rm = set_rtz();

    // perform square root in RTZ mode
    double result = std::sqrt(x);

    // load exceptions
    const auto fexps = load_exceptions();

    // reset rounding mode
    set_rm(rm);

    // check if overflow or underflow occurred
    FPY_ASSERT(
        !(fexps & (FE_OVERFLOW | FE_UNDERFLOW)),
        "sqrt: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & FE_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}

double fma(double x, double y, double z, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "fma: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    clear_exceptions();
    const int rm = set_rtz();

    // perform fused multiply-add in RTZ mode
    double result = std::fma(x, y, z);

    // load exceptions
    const auto fexps = load_exceptions();

    // reset rounding mode
    set_rm(rm);

    // check if overflow or underflow occurred
    FPY_ASSERT(
        !(fexps & (FE_OVERFLOW | FE_UNDERFLOW)),
        "fma: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & FE_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}

} // end namespace engine

} // end namespace fpy
