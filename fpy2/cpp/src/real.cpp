#include <cstring>

#include "fpy/real.hpp"
#include "fpy/params.hpp"

namespace fpy {

RealFloat::RealFloat(double x) {
    // format-dependent constants for double-precision floats
    using FP = ieee754_consts<11, 64>;

    // load floating-point data as unsigned integer
    uint64_t b;
    std::memcpy(&b, &x, sizeof(x));

    // decompose fields
    const uint64_t sbits = b & FP::SMASK;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // sign
    this->s = sbits != 0;

    // case split on exponent field
    if (ebits == 0) {
        // zero / subnormal
        this->exp = FP::EXPMIN;
        this->c = 0;
    } else if (ebits == FP::EONES) {
        // infinity or NaN
        FPY_ASSERT(false, "cannot convert infinity or NaN");
    } else {
        // normal
        this->exp = FP::EXPMIN + (ebits - 1);
        this->c = FP::IMPLICIT1 | mbits;
    }
}

RealFloat::RealFloat(float x) {
    // format-dependent constants for double-precision floats
    using FP = ieee754_consts<8, 32>;

    // load floating-point data as unsigned integer
    uint32_t b32;
    std::memcpy(&b32, &x, sizeof(x));

    // decompose fields
    uint64_t b = static_cast<uint64_t>(b32);
    const uint64_t sbits = b & FP::SMASK;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // sign
    this->s = sbits != 0;

    // case split on exponent field
    if (ebits == 0) {
        // zero / subnormal
        this->exp = FP::EXPMIN;
        this->c = 0;
    } else if (ebits == FP::EONES) {
        // infinity or NaN
        FPY_ASSERT(false, "cannot convert infinity or NaN");
    } else {
        // normal
        this->exp = FP::EXPMIN + (ebits - 1);
        this->c = FP::IMPLICIT1 | mbits;
    }
}

} // namespace fpy
