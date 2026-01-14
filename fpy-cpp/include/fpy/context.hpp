#pragma once

#include "types.hpp"

namespace fpy {

/// @brief Abstract base class for rounding contexts.
///
/// Rounding contexts encapsulate a rounding operation from real numbers
/// to a floating-point representation.
class Context {
public:
    virtual ~Context() = default;

    /// @brief Minimum precision using round-to-odd required for
    /// safe rerounding under this rounding context.
    virtual prec_t round_prec() const = 0;

    /// @brief Rounds `x` according to this rounding context.
    /// @param x a number to round
    /// @return the rounded number
    virtual double round(double x) const = 0;

    /// @brief Rounds `m * 2^exp` according to this rounding context.
    /// @param m integer significand
    /// @param exp base-2 exponent
    /// @return the rounded number
    virtual double round(int64_t m, exp_t exp) const = 0;
};

} // namespace fpy
