#pragma once

#include "types.hpp"

namespace fpy {

/// @brief Rounding modes for floating-point operations
/// 
/// When a real value is not representable in the target format,
/// rounding modes determine which floating-point value to choose.
enum class RoundingMode : uint8_t {
    RNE,               // Round to nearest, ties to even
    RNA,               // Round to nearest, ties away from zero
    RTP,               // Round toward +infinity (ceiling)
    RTN,               // Round toward -infinity (floor)
    RTZ,               // Round toward zero (truncation)
    RAZ,               // Round away from zero
    RTO,               // Round to odd
    RTE,               // Round to even
};

/// @brief Rounding direction
///
/// Indicates which value to round relative to the original value.
/// A `RoundingMode` can be mapped to a boolean indicating whether
/// the rounding is a nearest rounding and a `RoundingDirection`.
///
enum class RoundingDirection : uint8_t {
    TO_ZERO,
    AWAY_ZERO,
    TO_EVEN,
    TO_ODD,
};

//// @brief Returns whether the rounding mode is a nearest rounding mode.
inline bool is_nearest(RoundingMode mode) noexcept {
    return mode == RoundingMode::RNE || mode == RoundingMode::RNA;
}

/// @brief Returns the rounding direction for a given rounding mode and sign.
/// For nearest rounding modes, the direction is for tie-breaking.
RoundingDirection get_direction(RoundingMode mode, bool sign);

/// @brief Alias for `RoundingMode`.
using RM = RoundingMode;


/// @brief Summarary of rounding bits.
enum class RoundingBits : uint8_t {
    EXACT,          // representable: no rounding needed
    BELOW_HALFWAY,  // below the halfway point
    HALFWAY,        // exactly halfway
    ABOVE_HALFWAY   // above the halfway point
};

/// @brief Classifies the rounding bits based on the half and sticky bits.
RoundingBits to_rounding_bits(bool half_bit, bool sticky_bit) noexcept;

} // namespace fpy
