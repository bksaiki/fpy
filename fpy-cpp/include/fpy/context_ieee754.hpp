#pragma once

#include "context_mpb.hpp"
#include "params.hpp"
#include "types.hpp"

namespace fpy {

/// @brief IEEE 754 floating-point rounding context.
///
/// This context implements the usual IEEE 754 semantics for floating-point
/// arithmetic, including exponent bounds and overflow handling.
class IEEE754Context : public MPBContext {
private:
    /// @brief Number of exponent bits.
    prec_t es_;
    /// @brief Total number of bits (including sign bit).
    prec_t nbits_;

public:

    /// @brief Constructs an IEEE 754 context.
    /// @param es number of exponent bits
    /// @param nbits total number of bits (including sign bit)
    /// @param rm rounding mode
    IEEE754Context(prec_t es, prec_t nbits, RM rm);

    /// @brief Gets the number of exponent bits.
    inline prec_t es() const {
        return es_;
    }

    /// @brief Gets the total number of bits (including sign bit).
    inline prec_t nbits() const {
        return nbits_;
    }
};

} // namespace fpy
