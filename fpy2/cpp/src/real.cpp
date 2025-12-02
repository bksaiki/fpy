#include <cmath>
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
        this->c = mbits;
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

    // special case: zero
    if (c == 0) {
        return s ? -0.0 : 0.0;
    }

    // case split on normalization
    const auto [c, exp] = normalize_data(FP::P, FP::EXPMIN - 1);
    const prec_t p = std::bit_width(c);
    const uint64_t ebits = p == FP::P ? (exp - FP::EXPMIN + 1) : 0;
    const uint64_t mbits = c & bitmask<mant_t>(FP::M); 

    // cast to double
    const uint64_t sbits = s ? 1 : 0;
    const uint64_t b = (sbits << (FP::N - 1)) | (ebits << FP::M) | mbits;
    return std::bit_cast<double>(b);
}

std::pair<mant_t, exp_t> RealFloat::normalize_data(
    const std::optional<prec_t>& p,
    const std::optional<exp_t>& n
) const {
    // special case: zero
    if (c == 0) {
        if (n.has_value()) {
            return { 0, *n + 1 };
        } else {
            return { 0, this->exp };
        }
    }

    // case split on precision
    exp_t exp, shift;
    if (p.has_value()) {
        // requesting maximum precision
        shift = static_cast<exp_t>(*p) - static_cast<exp_t>(prec());
        exp = this->exp - shift;
        if (n.has_value() && exp <= *n) {
            // requesting lower bound on digits
            // exponent is too small, adjust if necessary
            const exp_t expmin = *n + 1;
            const exp_t adjust = expmin - exp;
            shift -= adjust;
            exp += adjust;
        }
    } else {
        // no maximum precision
        if (n.has_value()) {
            // requesting lower bound on digits
            exp = *n + 1;
            shift = this->exp - exp;
        } else {
            // no parameters specified, return copy
            exp = this->exp;
            shift = 0;
        }
    }

    // compute new significand `c`
    if (shift == 0) {
        // no shifting required
        return { c, exp };
    } else if (shift > 0) {
        // shift left by a non-negative amount
        const prec_t p = std::bit_width(c) + static_cast<prec_t>(shift);
        FPY_DEBUG_ASSERT(p <= 64, "normalize: precision exceeds 64 bits");
        return { c << shift, exp };
    } else {
        // shift right by a positive amount
        const mant_t c_lost = c & bitmask<mant_t>(-shift);
        FPY_DEBUG_ASSERT(c_lost == 0, "normalize: losing digits");
        return { c >> -shift, exp };
    }
}

RealFloat RealFloat::normalize(
    const std::optional<prec_t>& p,
    const std::optional<exp_t>& n
) const {
    // precision cannot exceed 64 bits
    FPY_DEBUG_ASSERT(
        !p.has_value() || *p <= 64,
        "normalize: precision exceeds 64 bits"
    );

    // compute normalized data
    const auto [c, exp] = normalize_data(p, n);
    return RealFloat(s, exp, c);
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
    const std::optional<prec_t>& max_p,
    const std::optional<exp_t>& min_n,
    RM rm
) const {
    // ensure one rounding parameter is specified
    FPY_ASSERT(
        max_p.has_value() || min_n.has_value(),
        "at least one parameter must be provided"
    );

    // compute the actual rounding parameters to be used
    const auto params = round_params(max_p, min_n, rm);

    // round
    return round_at(params);
}

RealFloat::round_params_t RealFloat::round_params(
    const std::optional<prec_t>& max_p,
    const std::optional<exp_t>& min_n,
    RM rm
) const {
    // case split on max_p
    if (max_p.has_value()) {
        // requesting maximum precision
        const auto p = *max_p;
        const exp_t n = e() - p;
        if (min_n.has_value()) {
            // requesting lower bound on digits
            // IEEE 754 style rounding
            return { p, std::max(*min_n, n), rm, true };
        } else {
            // no lower bound on digits
            // floating-point style rounding
            return { p, n, rm, true };
        }
    } else {
        // no maximum precision
        FPY_DEBUG_ASSERT(min_n.has_value(), "min_n must be specified if max_p is not");
        return { 0, *min_n, rm, false };
    }
}

RealFloat RealFloat::round_at(const RealFloat::round_params_t& params) const {
    // step 1. split the number at the rounding position
    auto [hi, lo] = split(params.n);

    // step 2. check if rounding was exact
    if (lo.is_zero()) {
        return hi;
    }

    // step 3. recover the rounding bits
    RoundingBits rb;
    if (lo.e() == params.n) {
        // the MSB of lo is at position n
        const bool half_bit = (lo.c >> (lo.prec() - 1)) != 0;
        const bool sticky_bit = (lo.c & bitmask<mant_t>(lo.prec() - 1)) != 0;
        rb = to_rounding_bits(half_bit, sticky_bit);
    } else {
        // the MSB of lo is below position n
        // half bit is 0, sticky bit is 1
        rb = RoundingBits::BELOW_HALFWAY;
    }

    // step 4. finalize rounding based on the rounding mode
    FPY_DEBUG_ASSERT(rb != RoundingBits::EXACT, "must be inexact here");
    hi.round_finalize(params, rb);

    return hi;
}

void RealFloat::round_finalize(const RealFloat::round_params_t& params, RoundingBits rb) {
    // increment if necessary
    if (round_increment(rb, params.rm)) {
        c += 1;
        if (params.has_p && prec() > params.p) {
            // adjust the exponent since we exceeded precision limit
            // the resulting value will be a power of two
            c >>= 1;
            exp += 1;
        }
    }

    // set inexact flag
    inexact = rb != RoundingBits::EXACT;
}

bool RealFloat::round_increment(RoundingBits rb, RM rm) const {
    // case split on nearest
    if (is_nearest(rm)) {
        // nearest rounding mode
        // case split on rounding bits
        switch (rb) {
            case RoundingBits::ABOVE_HALFWAY:
                // above halfway
                return true;
            case RoundingBits::HALFWAY:
                // exact halfway
                return round_direction(rm);
            case RoundingBits::BELOW_HALFWAY:
            case RoundingBits::EXACT:
                // below halfway or exact
                return false;
            default:
                FPY_UNREACHABLE();
        }
    } else {
        // non-nearest rounding mode
        if (rb != RoundingBits::EXACT) {
            // inexact
            return round_direction(rm);
        } else {
            // exact
            return false;
        }
    }
}

bool RealFloat::round_direction(RM rm) const {
    switch (get_direction(rm, s)) {
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

} // namespace fpy
