#pragma once

#include "context.hpp"
#include "round_opt.hpp"
#include "types.hpp"

namespace fpy {

/// @brief MPFR-style floating-point rounding context with minimum exponent.
///
/// This context represents floating-point values with an arbitrary precision,
/// a minimum exponent bound, and a specified rounding mode. Subnormalization
/// is computed as n = emin - p.
class MPSContext : public Context {
private:
    /// @brief Maximum precision of this context.
    prec_t prec_;
    /// @brief Minimum exponent of this context.
    exp_t emin_;
    /// @brief Rounding mode of this context.
    RM rm_;
    /// @brief Precomputed subnormalization parameter (n = emin - p).
    exp_t n_;

public:

    MPSContext(prec_t prec, exp_t emin, RM rm) :
        prec_(prec), emin_(emin), rm_(rm), n_(emin - static_cast<exp_t>(prec)) {}

    /// @brief Gets the maximum precision of this context.
    inline prec_t prec() const {
        return prec_;
    }

    /// @brief Gets the minimum exponent of this context.
    inline exp_t emin() const {
        return emin_;
    }

    /// @brief Gets the rounding mode of this context.
    inline RM rm() const {
        return rm_;
    }

    /// @brief Gets the precomputed subnormalization parameter (n = emin - p).
    inline exp_t n() const {
        return n_;
    }

    inline prec_t round_prec() const override {
        return prec_;
    }

    inline double round(double x) const override {
        return round_opt::round(x, prec_, n_, rm_);
    }
};

} // namespace fpy
