#include <bit>

#include "fpy/context_ieee754.hpp"

namespace fpy {

static prec_t ieee754_prec(prec_t es, prec_t nbits) {
    return nbits - es;
}

static exp_t ieee754_emax(prec_t es) {
    return static_cast<exp_t>(bitmask<uint32_t>(es - 1));
}

static exp_t ieee754_emin(prec_t es) {
    return 1 - ieee754_emax(es);
}

static double ieee754_max_value(prec_t es, prec_t nbits) {
    using FP64 = ieee754_consts<11, 64>; // IEEE 754 double precision

    // format parameters
    const prec_t prec = ieee754_prec(es, nbits);
    const exp_t emax = ieee754_emax(es);

    // encode maximum value as double
    const uint64_t mbits = bitmask<mant_t>(prec - 1) << (FP64::P - prec);
    const uint64_t ebits = static_cast<uint64_t>(emax + FP64::BIAS) << FP64::M;
    const uint64_t bits = ebits | mbits;
    return std::bit_cast<double>(bits);
}

IEEE754Context::IEEE754Context(prec_t es, prec_t nbits, RM rm)
    : MPBContext(
        ieee754_prec(es, nbits),
        ieee754_emin(es), rm,
        ieee754_max_value(es, nbits)),
    es_(es), nbits_(nbits) {}

} // namespace fpy
