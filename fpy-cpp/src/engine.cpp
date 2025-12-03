#include <bit>
#include <cmath>

#include "fpy/arch.hpp"
#include "fpy/engine.hpp"

namespace fpy {

namespace engine {

static double finalize(double result, int fexps) {
    // check if overflow or underflow occurred
    FPY_ASSERT(
        !(fexps & (arch::EXCEPT_OVERFLOW | arch::EXCEPT_UNDERFLOW)),
        "rto_add: overflow or underflow occurred"
    );

    // check inexactness
    if (fexps & arch::EXCEPT_INEXACT) {
        uint64_t b = std::bit_cast<uint64_t>(result);
        b |= 1; // set LSB
        result = std::bit_cast<double>(b);
    }

    return result;
}


double add(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "rto_add: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform addition in RTZ mode
    double result = x + y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

double sub(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "sub: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform subtraction in RTZ mode
    double result = x - y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

double mul(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "mul: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform multiplication in RTZ mode
    double result = x * y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

double div(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "div: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform division in RTZ mode
    double result = x / y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

double sqrt(double x, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "sqrt: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform square root in RTZ mode
    double result = std::sqrt(x);

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

double fma(double x, double y, double z, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "fma: requested precision exceeds double-precision capability"
    );

    // prepare floating-point environment
    const auto old_mode = arch::prepare_rto();

    // perform fused multiply-add in RTZ mode
    double result = std::fma(x, y, z);

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_mode);

    // finalize result
    return finalize(result, fexps);
}

} // end namespace engine

} // end namespace fpy
