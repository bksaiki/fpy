#pragma once

#include "context.hpp"
#include "round_opt.hpp"
#include "types.hpp"

namespace fpy {

/// @brief MPFR-style floating-point rounding context.
///
/// This context represents floating-point values with an arbitrary precision,
/// no exponent bounds, and a specified rounding mode.
class MPContext : public Context {
private:
    /// @brief Maximum precision of this context.
    prec_t prec_;
    /// @brief Rounding mode of this context.
    RM rm_;

public:

    MPContext(prec_t prec, RM rm) : prec_(prec), rm_(rm) {}

    /// @brief Gets the maximum precision of this context.
    inline prec_t prec() const {
        return prec_;
    }

    /// @brief Gets the rounding mode of this context.
    inline RM rm() const {
        return rm_;
    }

    inline prec_t round_prec() const override {
        return prec_ + 2;
    }

    inline double round(double x) const override {
        return round_opt::round(x, prec_, std::nullopt, rm_);
    }

    inline double round(int64_t m, exp_t exp) const override {
        return round_opt::round(m, exp, prec_, std::nullopt, rm_);
    }
};

} // namespace fpy
