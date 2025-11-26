#pragma once

namespace fpy {

// NOTE: Based on FPy's RoundingMode and RoundingDirection
// Adjust these enums to match the subset you need

/**
 * @brief Rounding modes for floating-point operations
 * 
 * NOTE: Add/remove modes based on your requirements
 */
enum class RoundingMode {
    NEAREST_EVEN,      // Round to nearest, ties to even
    NEAREST_AWAY,      // Round to nearest, ties away from zero
    TOWARD_ZERO,       // Truncate (round toward zero)
    TOWARD_POSITIVE,   // Round toward +infinity (ceiling)
    TOWARD_NEGATIVE,   // Round toward -infinity (floor)
    // NOTE: Add other rounding modes as needed
};

/**
 * @brief Rounding direction (simplified)
 */
enum class RoundingDirection {
    DOWN,
    UP,
    NEAREST,
};

/**
 * @brief Overflow handling modes
 * 
 * NOTE: Define how to handle overflow conditions
 */
enum class OverflowMode {
    SATURATE,   // Saturate to max/min representable value
    INFINITY,   // Return infinity
    WRAP,       // Wrap around
    // NOTE: Add other overflow modes as needed
};

// NOTE: Add utility functions for rounding operations
// Example:
// RoundingDirection get_rounding_direction(RoundingMode mode, bool sign);

} // namespace fpy
