#include <cstring>

#include "fpy/real.hpp"
#include "fpy/params.hpp"

namespace fpy {

RealFloat::RealFloat(double x) {
    // format-dependent constants for double-precision floats
    using FP = ieee754_consts<11, 64>;

    // load floating-point data as unsigned integer
    uint64_t b = std::bit_cast<uint64_t>(x);

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

    // flag data
    this->inexact = false;
}

RealFloat::RealFloat(float x) {
    // format-dependent constants for double-precision floats
    using FP = ieee754_consts<8, 32>;

    // load floating-point data as unsigned integer
    uint64_t b = static_cast<uint64_t>(std::bit_cast<uint32_t>(x));

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

    // flag data
    this->inexact = false;
}

RealFloat::operator double() const {
    // format-dependent constants for double-precision floats
    using FP = ieee754_consts<11, 64>;

    // handle zero
    if (c == 0) {
        return s ? -0.0 : 0.0;
    }

    // normalize the significand to get actual exponent
    const exp_t actual_exp = exp + prec() - 1;

    // check for overflow (exponent too large)
    if (actual_exp > FP::EXPMAX) {
        FPY_ASSERT(false, "cannot convert: overflow to infinity");
    }

    // check for underflow (exponent too small)
    if (actual_exp < FP::EXPMIN) {
        FPY_ASSERT(false, "cannot convert: underflow to zero/subnormal");
    }

    // compute biased exponent
    const uint64_t ebits = static_cast<uint64_t>(actual_exp - FP::EXPMIN + 1);

    // normalize mantissa to have implicit leading 1
    // shift to align with mantissa field width
    const prec_t p = prec();
    uint64_t mbits;
    
    if (p > FP::M + 1) {
        // too many bits, need to truncate
        FPY_ASSERT(false, "cannot convert: precision loss");
    } else if (p == FP::M + 1) {
        // exact fit, remove implicit 1
        mbits = c & FP::MMASK;
    } else {
        // need to shift left
        mbits = (c << (FP::M + 1 - p)) & FP::MMASK;
    }

    // construct the bit pattern
    const uint64_t sbits = s ? FP::SMASK : 0;
    const uint64_t b = sbits | (ebits << FP::M) | mbits;

    return std::bit_cast<double>(b);
}

std::tuple<RealFloat, RealFloat> RealFloat::split(exp_t n) const {
    if (c == 0) {
        // special case: 0
        const RealFloat hi(s, n + 1, 0);
        const RealFloat lo(s, n, 0);
        return { hi, lo };
    } else if (n >= e()) {
        // all digits are in the lower part
        const RealFloat hi(s, n + 1, 0);
        return { hi, *this };
    } else if (n < exp) {
        // all digits are in the upper part
        const RealFloat lo(s, n, 0);
        return { *this, lo };
    } else {
        // splitting the digits

        // length of lower part
        const prec_t p_lo = (n + 1) - exp;
        const auto mask_lo = bitmask<mant_t>(p_lo);

        // exponents
        const exp_t exp_hi = exp + p_lo;
        const exp_t exp_lo = exp;

        // mantissas
        const mant_t c_hi = c >> p_lo;
        const mant_t c_lo = c & mask_lo;

        const RealFloat hi(s, exp_hi, c_hi);
        const RealFloat lo(s, exp_lo, c_lo);
        return { hi, lo };
    }
}

RealFloat RealFloat::round(
    std::optional<prec_t> max_p,
    std::optional<exp_t> min_n,
    RM rm
) const {
    // ensure one rounding parameter is specified
    FPY_ASSERT(
        max_p.has_value() || min_n.has_value(),
        "at least one parameter must be provided"
    );

    // compute the actual rounding parameters to be used
    auto [p, n] = round_params(max_p, min_n);

    // round
    return round_at(p, n, rm);
}

std::tuple<std::optional<prec_t>, exp_t> RealFloat::round_params(
    std::optional<prec_t> max_p,
    std::optional<exp_t> min_n
) const {
    // case split on max_p
    if (max_p.has_value()) {
        // requesting maximum precision
        const auto p = max_p.value();
        if (min_n.has_value()) {
            // requesting lower bound on digits
            // IEEE 754 style rounding
            const exp_t n = std::max(min_n.value(), static_cast<exp_t>(e() - p));
            return { p, n };
        } else {
            // no lower bound on digits
            // floating-point style rounding
            const exp_t n = e() - p;
            return { p, n };
        }
    } else {
        // no maximum precision
        FPY_ASSERT(min_n.has_value(), "min_n must be specified if max_p is not");
        const exp_t n = min_n.value();
        return { std::optional<prec_t>(), n };
    }
}

RealFloat RealFloat::round_at(
    std::optional<prec_t> p, exp_t n, RM rm
) const {
    // step 1. split the number at the rounding position
    auto [hi, lo] = split(n);

    // step 2. check if rounding was exact
    if (lo.is_zero()) {
        return hi;
    }

    // step 3. recover the rounding bits
    bool half_bit, sticky_bit;
    if (lo.e() == n) {
        // the MSB of lo is at position n
        half_bit = (lo.c >> (lo.prec() - 1)) != 0;
        sticky_bit = (lo.c & bitmask<mant_t>(lo.prec() - 1)) != 0;
    } else {
        // the MSB of lo is below position n
        half_bit = false;
        sticky_bit = true;
    }

    // step 4. finalize rounding based on the rounding mode
    hi.round_finalize(half_bit, sticky_bit, p, rm);

    return hi;
}

void RealFloat::round_finalize(
    bool half_bit,
    bool sticky_bit,
    std::optional<prec_t> p,
    RM rm
) {
    // prepare the rounding operation
    bool increment = round_direction(half_bit, sticky_bit, rm);

    // increment if necessary
    if (increment) {
        c += 1;
        if (p.has_value() && prec() > p.value()) {
            // adjust the exponent since we exceeded precision limit
            // the resulting value will be a power of two
            c >>= 1;
            exp += 1;
        }
    }

    // set inexact flag
    inexact = half_bit || sticky_bit;
}

bool RealFloat::round_direction(bool half_bit, bool sticky_bit, RM rm) const {
    // convert the rounding mode to a direction
    auto [nearest, direction] = to_direction(rm, s);

    // case split on nearest
    if (nearest) {
        // nearest rounding mode
        // case split on halfway bit
        if (half_bit) {
            // at least halfway
            if (sticky_bit) {
                // above halfway
                return true;
            } else {
                // exact halfway
                switch (direction) {
                    case RoundingDirection::TO_ZERO:
                        return false;
                    case RoundingDirection::AWAY_ZERO:
                        return true;
                    case RoundingDirection::TO_EVEN:
                        return (c & 1) != 0;
                    case RoundingDirection::TO_ODD:
                        return (c & 1) == 0;
                    default:
                        FPY_UNREACHABLE();
                }
            }
        } else {
            // below halfway
            return false;
        }
    } else {
        // non-nearest rounding mode
        if (half_bit || sticky_bit) {
            // inexact
            switch (direction) {
                case RoundingDirection::TO_ZERO:
                    return false;
                case RoundingDirection::AWAY_ZERO:
                    return true;
                case RoundingDirection::TO_EVEN:
                    return (c & 1) != 0;
                case RoundingDirection::TO_ODD:
                    return (c & 1) == 0;
                default:
                    FPY_UNREACHABLE();
            }
        } else {
            // exact
            return false;
        }
    }
}

} // namespace fpy
