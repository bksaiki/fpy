#include <cmath>

#include "fpy/context_mpb.hpp"
#include "fpy/params.hpp"
#include "fpy/round.hpp"

namespace fpy {

/// @brief Should overflow round to infinity?
/// @param s Sign of the unbounded result
/// @return Whether to round to infinity
static bool overflow_to_infinity(RM rm, bool s, bool maxval_odd) {
    const auto dir = get_direction(rm, s);
    switch (dir)
    {
    case RoundingDirection::TO_ZERO:
        // always round towards zero
        return false;
    case RoundingDirection::AWAY_ZERO:
        // always round away from zero
        return true;
    case RoundingDirection::TO_EVEN:
        // round to infinity if maxval is odd
        return maxval_odd;
    case RoundingDirection::TO_ODD:
        // round to infinity if maxval is even
        return !maxval_odd;
    default:
        FPY_UNREACHABLE("invalid rounding direction");
    }
}

/// @brief Apply overflow handling to an already-rounded value.
/// @param x the rounded value from the underlying MPS context
/// @param rm rounding mode
/// @param maxval maximum representable magnitude
/// @param maxval_odd whether maxval has an odd mantissa
/// @return the value with overflow handling applied
static double round_overflow(double x, RM rm, double maxval, bool maxval_odd) {
    // Handle special cases first
    if (!std::isfinite(x)) {
        return x; // NaN or exact infinity
    }

    // Check for overflow
    if (std::abs(x) > maxval) {
        // should we round to infinity?
        const bool s = std::signbit(x);
        if (overflow_to_infinity(rm, s, maxval_odd)) {
            // round to infinity
            static constexpr double inf = std::numeric_limits<double>::infinity();
            return std::copysign(inf, x);
        } else {
            // round to maxval
            return std::copysign(maxval, x);
        }
    }

    return x;
}

MPBContext::MPBContext(prec_t prec, exp_t emin, RM rm, double maxval) 
    : mps_ctx_(prec, emin, rm), maxval_(maxval) {
    using FP = ieee754_consts<11, 64>; // IEEE 754 double precision

    // check that the maximum value is valid
    FPY_ASSERT(std::isfinite(maxval_), "maxval must be finite");
    FPY_ASSERT(maxval_ == mps_ctx_.round(maxval_), "maxval must be exactly representable in this context");

    // check if the maximum value is odd
    const uint64_t bits = std::bit_cast<uint64_t>(maxval_);
    const int pth_bit_pos = static_cast<int>(FP::M) - static_cast<int>(prec) + 1;
    maxval_odd_ = (pth_bit_pos >= 0) && ((bits >> pth_bit_pos) & 1);
}


double MPBContext::round(double x) const {
    // round without overflow handling
    x = mps_ctx_.round(x);
    
    // apply overflow handling
    return round_overflow(x, mps_ctx_.rm(), maxval_, maxval_odd_);
}

double MPBContext::round(int64_t m, exp_t exp) const {
    // round without overflow handling
    double x = mps_ctx_.round(m, exp);
    
    // apply overflow handling
    return round_overflow(x, mps_ctx_.rm(), maxval_, maxval_odd_);
}

} // namespace fpy
