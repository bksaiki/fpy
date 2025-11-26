#pragma once

#include <bit>

#include "types.hpp"

namespace fpy {

/// @brief Floating-point type encoding finite values.
///
/// This is a number of the form `(-1)^s * c * 2^exp` where
/// `c` is a non-negative integer and `exp` is an integer.
class RealFloat {

public:

    exp_t exp;
    mant_t c;
    bool s;

    /// @brief default constructor: constructs +0
    explicit RealFloat() {
        this->s = false;
        this->exp = 0;
        this->c = 0;
    }

    /// @brief constructs a `RealFloat` from the triple `(s, exp, c)`
    explicit RealFloat(bool s, exp_t exp, mant_t c) {
        this->s = s;
        this->exp = exp;
        this->c = c;
    }

    /// @brief constructs a `RealFloat` from a `double`.
    explicit RealFloat(double x);

    /// @brief constructs a `RealFloat` from a `float`.
    explicit RealFloat(float x);

    /// @brief the precision of the significand.
    inline prec_t prec() const {
        return std::bit_width(c);
    }

    /// @brief the normalized exponent of this number.
    /// If `this->is_zero()` then this method returns `this->exp - 1`.
    inline exp_t e() const {
        return exp + prec() - 1;
    }

    /// @brief the first unrepresentable digit below the significant digits.
    /// This is always `this->exp - 1`.
    inline exp_t n() const {
        return exp - 1;
    }
};


} // namespace fpy
