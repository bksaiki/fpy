#include <bit>
#include <cmath>

#include "fpy/arch.hpp"
#include "fpy/engine.hpp"

namespace fpy {

namespace engine {

static double finalize(double result, unsigned int fexps) {
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
    const auto old_csr = arch::prepare_rto();

    // perform addition in RTZ mode
    double result = x + y;

    // load exceptions and reset rounding mode
    const auto fexps = arch::rto_status(old_csr);

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

double add_exact(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "add_exact: requested precision exceeds double-precision capability"
    );

#if defined(FPY_DEBUG)
    // prepare floating-point environment
    arch::clear_exceptions();
#endif

    // perform exact addition
    double result = x + y;

#if defined(FPY_DEBUG)
    // check for inexactness or overflow
    const auto fexps = arch::get_exceptions();
    FPY_ASSERT(
        !(fexps & (arch::EXCEPT_INEXACT | arch::EXCEPT_OVERFLOW)),
        "add_exact: addition was not exact"
    );
#endif

    // return result
    return result;
}

double sub_exact(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "sub_exact: requested precision exceeds double-precision capability"
    );

#if defined(FPY_DEBUG)
    // prepare floating-point environment
    arch::clear_exceptions();
#endif

    // perform exact subtraction
    double result = x - y;

#if defined(FPY_DEBUG)
    // check for inexactness or overflow
    const auto fexps = arch::get_exceptions();
    FPY_ASSERT(
        !(fexps & (arch::EXCEPT_INEXACT | arch::EXCEPT_OVERFLOW)),
        "sub_exact: subtraction was not exact"
    );
#endif

    // return result
    return result;
}

double mul_exact(double x, double y, prec_t p) {
    // double-precision only guarantees 53 bits of precision
    FPY_ASSERT(
        p <= 53,
        "mul_exact: requested precision exceeds double-precision capability"
    );

#if defined(FPY_DEBUG)
    // prepare floating-point environment
    arch::clear_exceptions();
#endif

    // perform exact multiplication
    double result = x * y;

#if defined(FPY_DEBUG)
    // check for inexactness or overflow
    const auto fexps = arch::get_exceptions();
    FPY_ASSERT(
        !(fexps & (arch::EXCEPT_INEXACT | arch::EXCEPT_OVERFLOW)),
        "mul_exact: multiplication was not exact"
    );
#endif

    // return result
    return result;
}



} // end namespace engine

} // end namespace fpy
