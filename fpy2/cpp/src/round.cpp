#include "fpy/round.hpp"
#include "fpy/utils.hpp"

namespace fpy {

std::tuple<bool, RoundingDirection> to_direction(RoundingMode mode, bool sign) {
    switch (mode) {
        case RoundingMode::RNE:
            return { true, RoundingDirection::TO_EVEN };
        case RoundingMode::RNA:
            return { true, RoundingDirection::AWAY_ZERO };
        case RoundingMode::RTP:
            return { false, sign ? RoundingDirection::TO_ZERO : RoundingDirection::AWAY_ZERO };
        case RoundingMode::RTN:
            return { false, sign ? RoundingDirection::AWAY_ZERO : RoundingDirection::TO_ZERO };
        case RoundingMode::RTZ:
            return { false, RoundingDirection::TO_ZERO };
        case RoundingMode::RAZ:
            return { false, RoundingDirection::AWAY_ZERO };
        case RoundingMode::RTO:
            return { false, RoundingDirection::TO_ODD };
        case RoundingMode::RTE:
            return { false, RoundingDirection::TO_EVEN };
        default:
            FPY_UNREACHABLE("invalid rounding mode");
    }
}

} // namespace fpy
