#pragma once

#include <cstdint>

#include "utils.hpp"

namespace fpy {

/// @brief floating-point exponent
using exp_t = int32_t;

/// @brief container type for a floating-point mantissa
using mant_t = uint64_t;

/// @brief container type for precision
using prec_t = uint64_t;


/// @brief Generates a bitmask of length `k` for type `T`.
/// @tparam T an unsigned integral type 
/// @param k size of the bitmask
/// @return the bitmask
template <typename T>
T bitmask(uint64_t k) {
    FPY_STATIC_ASSERT(std::is_integral_v<T> && std::is_unsigned_v<T>, "T must be an unsigned integer");
    FPY_DEBUG_ASSERT(k < 64, "exceeded maximum bitmask size");
    FPY_DEBUG_ASSERT(k <= 8 * sizeof(T), "exceeded maximum bitmask size for T");
    return static_cast<T>((1ULL << k) - 1);
}

} // namespace fpy
