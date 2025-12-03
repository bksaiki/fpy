#include "fpy/round.hpp"
#include "fpy/utils.hpp"

namespace fpy {

RoundingDirection get_direction(RoundingMode mode, bool sign) {
    switch (mode) {
        case RoundingMode::RNE:
            return RoundingDirection::TO_EVEN;
        case RoundingMode::RNA:
            return RoundingDirection::AWAY_ZERO;
        case RoundingMode::RTP:
            return sign ? RoundingDirection::TO_ZERO : RoundingDirection::AWAY_ZERO;
        case RoundingMode::RTN:
            return sign ? RoundingDirection::AWAY_ZERO : RoundingDirection::TO_ZERO;
        case RoundingMode::RTZ:
            return RoundingDirection::TO_ZERO;
        case RoundingMode::RAZ:
            return RoundingDirection::AWAY_ZERO;
        case RoundingMode::RTO:
            return RoundingDirection::TO_ODD;
        case RoundingMode::RTE:
            return RoundingDirection::TO_EVEN;
        default:
            FPY_UNREACHABLE("invalid rounding mode");
    }
}

RoundingBits to_rounding_bits(bool half_bit, bool sticky_bit) noexcept {
    if (half_bit) {
        if (sticky_bit) {
            return RoundingBits::ABOVE_HALFWAY;
        } else {
            return RoundingBits::HALFWAY;
        }
    } else {
        if (sticky_bit) {
            return RoundingBits::BELOW_HALFWAY;
        } else {
            return RoundingBits::EXACT;
        }
    }
}

} // namespace fpy
