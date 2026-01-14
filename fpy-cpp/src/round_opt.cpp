#include <cmath>

#include "fpy/params.hpp"
#include "fpy/round_opt.hpp"

namespace fpy {

namespace round_opt {

double round(double x, prec_t p, const std::optional<exp_t>& n, RM rm) {
    using FP = ieee754_consts<11, 64>; // double precision

    // Fast path: special values (infinity, NaN, zero)
    if (!std::isfinite(x) || x == 0.0) {
        return x;
    }

    // load floating-point data as integer
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const bool s = (b >> (FP::N - 1)) != 0;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // decode floating-point data
    exp_t e;
    mant_t c;
    if (UNLIKELY(ebits == 0)) {
        // subnormal
        const auto lz = FP::P - std::bit_width(mbits);
        e = FP::EMIN - lz;
        c = mbits << lz;
    } else {
        // normal (assuming no infinity or NaN)
        e = static_cast<exp_t>(ebits) - FP::BIAS;
        c = FP::IMPLICIT1 | mbits;
    }

    // finalize rounding (mantissa has precision `FP::P`)
    return __round_finalize<FP::P>(s, e, c, p, n, rm);
}

double round(int64_t m, exp_t exp, prec_t p, const std::optional<exp_t>& n, RM rm) {
    static constexpr int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
    static constexpr prec_t PREC = 63;

    // Fast path: zero
    if (m == 0) {
        return 0.0;
    }

    // Decode `m` into sign-magnitude
    bool s;
    mant_t c;
    if (m == MIN_VAL) {
        // special decode to ensure 63 bits of precision
        s = true;
        c = 1ULL << (PREC - 1);
        exp += 1;
    } else if (m < 0) {
        s = true;
        c = static_cast<mant_t>(std::abs(m));
    } else {
        s = false;
        c = static_cast<mant_t>(m);
    }

    // we may have less precision than expected
    // guaranteed to have at most 63 bits
    const auto lz = PREC - std::bit_width(c);
    c <<= lz;
    exp -= lz;

    // calculate normalized exponent
    const exp_t e = exp + (PREC - 1);

    // finalize rounding (mantissa has precision 63)
    return __round_finalize<63>(s, e, c, p, n, rm);
}

} // namespace round_opt

} // namespace fpy
