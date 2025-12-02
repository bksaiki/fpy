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

    // numerical data (ordered for optimal packing)
    mant_t c = 0;         // 8 bytes
    exp_t exp = 0;        // 8 bytes
    bool s = false;       // 1 byte
    bool inexact = false; // 1 byte
                          // 6 bytes padding

    /// @brief default constructor: constructs +0
    explicit constexpr RealFloat() noexcept = default;

    /// @brief constructs a `RealFloat` from the triple `(s, exp, c)`
    explicit constexpr RealFloat(bool s, exp_t exp, mant_t c) noexcept : c(c), exp(exp), s(s) {}

    /// @brief constructs a `RealFloat` from a `double`.
    explicit RealFloat(double x);

    /// @brief constructs a `RealFloat` from a `float`.
    explicit RealFloat(float x);

    /// @brief converts a `RealFloat` to a `double`.
    explicit operator double() const;

    /// @brief Represents 0?
    inline bool is_zero() const noexcept {
        return c == 0;
    }

    /// @brief Represents a positive number?
    inline bool is_positive() const noexcept {
        return c != 0 && !s;
    }

    /// @brief Represents a negative number?
    inline bool is_negative() const noexcept {
        return c != 0 && s;
    }

    /// @brief the precision of the significand.
    inline prec_t prec() const noexcept {
        return std::bit_width(c);
    }

    /// @brief the normalized exponent of this number.
    /// If `this->is_zero()` then this method returns `this->exp - 1`.
    inline exp_t e() const noexcept {
        return exp + prec() - 1;
    }

    /// @brief the first unrepresentable digit below the significant digits.
    /// This is always `this->exp - 1`.
    inline exp_t n() const noexcept {
        return exp - 1;
    }

    /// @brief Returns a value numerically equivalent to this number,
    /// according to the precision `p` and position `n`.
    /// If this number is non-zero:
    ///
    /// - `std::nullopt, std::nullopt`: a copy of this number.
    ///
    /// - `p, std::nullopt`: the result has exactly `p` significant digits.
    ///
    /// - `std::nullopt, n`: the result with `exp == n + 1`.
    ///
    /// - `p, n`: the result has maximal precision up to `p` digits
    ///  such that `exp >= n + 1`.
    RealFloat normalize(const std::optional<prec_t>& p, const std::optional<exp_t>& n) const;

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
    RealFloat round(const std::optional<prec_t>& p, const std::optional<exp_t>& n, RM rm) const;

private:

    /// @brief Computes the actual rounding parameters `p` and `n`
    /// based on requested rounding parameters `max_p` and `min_n`.
    std::tuple<std::optional<prec_t>, exp_t> round_params(
        const std::optional<prec_t>& max_p,
        const std::optional<exp_t>& min_n
    ) const;

    /// @brief Rounds this value based on the rounding parameters `p` and `n`.
    RealFloat round_at(const std::optional<prec_t>& p, exp_t n, RM rm) const;

    /// @brief Finalizes rounding of this number based on rounding digits
    /// and rounding mode. This operation mutates the number.
    void round_finalize(
        const std::optional<prec_t>& p,
        bool half_bit,
        bool sticky_bit,
        RM rm
    );

    /// @brief Determines the direction to round based on the rounding mode.
    bool round_direction(bool half_bit, bool sticky_bit, RM rm) const;
};


} // namespace fpy
