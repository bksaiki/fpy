#pragma once

#include <bit>
#include <optional>
#include <tuple>

#include "round.hpp"
#include "types.hpp"

namespace fpy {

/// @brief Floating-point type encoding finite values.
///
/// This is a number of the form `(-1)^s * c * 2^exp` where
/// `c` is a non-negative integer and `exp` is an integer.
class RealFloat {

public:

    // numerical data
    exp_t exp;
    mant_t c;
    bool s;

    // flag data
    bool inexact;

    /// @brief default constructor: constructs +0
    explicit RealFloat() {
        this->s = false;
        this->exp = 0;
        this->c = 0;
        this->inexact = false;
    }

    /// @brief constructs a `RealFloat` from the triple `(s, exp, c)`
    explicit RealFloat(bool s, exp_t exp, mant_t c) {
        this->s = s;
        this->exp = exp;
        this->c = c;
        this->inexact = false;
    }

    /// @brief constructs a `RealFloat` from a `double`.
    explicit RealFloat(double x);

    /// @brief constructs a `RealFloat` from a `float`.
    explicit RealFloat(float x);

    /// @brief Represents 0?
    inline bool is_zero() const {
        return c == 0;
    }

    /// @brief Represents a positive number?
    inline bool is_positive() const {
        return c != 0 && !s;
    }

    /// @brief Represents a negative number?
    inline bool is_negative() const {
        return c != 0 && s;
    }

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

    /// @brief Splits this number into two values based on
    /// a digit position `n`.
    /// 
    /// The first value has the digits that are more significant
    /// than the digit position `n`. The second value has the digits
    /// that are at or below `n`.
    std::tuple<RealFloat, RealFloat> split(exp_t n) const;

    /// @brief Rounds this number to at most `max_p` digits of position
    /// or a least absolute digit posiiton `min_n`, whichever bound is
    /// encountered first. At least one of `max_p` or `min_n` must
    /// be specified.
    ///
    /// If only `min_n` is given, rounding is perfomed like fixed-point rounding.
    /// If only `max_p` is given, rounding is performed like floating-point
    /// without an exponent bound; the integer significand has at most `max_p` digits.
    /// If both are specified, the rounding is performed like IEEE 754
    /// floating-point arithmetic.
    RealFloat round(std::optional<prec_t> p, std::optional<exp_t> n, RM rm) const;

private:

    /// @brief Computes the actual rounding parameters `p` and `n`
    /// based on requested rounding parameters `max_p` and `min_n`.
    std::tuple<std::optional<prec_t>, exp_t> round_params(
        std::optional<prec_t> max_p,
        std::optional<exp_t> min_n
    ) const;

    /// @brief Rounds this value based on the rounding parameters `p` and `n`.
    RealFloat round_at(std::optional<prec_t> p, exp_t n, RM rm) const;

    /// @brief Finalizes rounding of this number based on rounding digits
    /// and rounding mode. This operation mutates the number.
    void round_finalize(
        bool half_bit,
        bool sticky_bit,
        std::optional<prec_t> p,
        RM rm
    );

    /// @brief Determines the direction to round based on the rounding mode.
    bool round_direction(bool half_bit, bool sticky_bit, RM rm) const;
};


} // namespace fpy
