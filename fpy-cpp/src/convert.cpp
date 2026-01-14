#include <cmath>

#include "fpy/convert.hpp"
#include "fpy/params.hpp"

namespace fpy {

std::tuple<int64_t, exp_t> to_fixed(double x) {
    using FP = ieee754_consts<11, 64>; // double precision
    FPY_ASSERT(std::isfinite(x), "to_fixed: input must be finite");

    // load floating-point data as integer
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const bool s = (b >> (FP::N - 1)) != 0;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // decode (unsigned) floating-point data
    exp_t exp; // unnormalized exponent
    mant_t c;  // integer significand
    if (ebits == 0) {
        // zero or subnormal
        exp = FP::EXPMIN;
        c = mbits;
    } else {
        // normal (assuming no infinity or NaN)
        const exp_t e = static_cast<exp_t>(ebits) - FP::BIAS;
        exp = e - static_cast<exp_t>(FP::M);
        c = FP::IMPLICIT1 | mbits;
    }

    // minimize the precision of the significand
    const auto tz = std::countr_zero(c);
    c >>= tz;
    exp += static_cast<exp_t>(tz);

    // apply sign
    const int64_t m = s ? -static_cast<int64_t>(c) : static_cast<int64_t>(c);
    return std::make_tuple(m, exp);
}


} // end namespace fpy
