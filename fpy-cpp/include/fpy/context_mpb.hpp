#pragma once

#include <cmath>

#include "context_mps.hpp"
#include "types.hpp"

namespace fpy {

/// @brief MPFR-style floating-point rounding context with minimum exponent and overflow.
///
/// This context represents floating-point values with an arbitrary precision,
/// a minimum exponent bound, a specified rounding mode, and a maximum representable
/// magnitude. When a value exceeds this maximum magnitude, it is treated as an overflow.
class MPBContext : public Context {
private:
    /// @brief Underlying MPS context for subnormalization handling.
    MPSContext mps_ctx_;
    /// @brief The overflow value (maximum representable magnitude).
    double maxval_;
    /// @brief Is the maximum value odd?
    bool maxval_odd_;

public:

    explicit MPBContext(prec_t prec, exp_t emin, RM rm, double maxval);

    /// @brief Gets the maximum precision of this context.
    inline prec_t prec() const {
        return mps_ctx_.prec();
    }

    /// @brief Gets the minimum exponent of this context.
    inline exp_t emin() const {
        return mps_ctx_.emin();
    }

    /// @brief Gets the maximum exponent of this context.
    inline exp_t emax() const {
        return maxval_ == 0.0 ? 0 : std::logb(maxval_);
    }

    /// @brief Gets the rounding mode of this context.
    inline RM rm() const {
        return mps_ctx_.rm();
    }

    /// @brief Gets the precomputed subnormalization parameter (n = emin - p).
    inline exp_t n() const {
        return mps_ctx_.n();
    }

    /// @brief This context without overflow handling.
    inline const MPSContext& mps_context() const {
        return mps_ctx_;
    }

    /// @brief Gets the overflow value (maximum representable magnitude).
    inline double maxval() const {
        return maxval_;
    }

    /// @brief Returns true if maxval is odd (pth bit of mantissa is set).
    inline bool maxval_odd() const {
        return maxval_odd_;
    }

    inline prec_t round_prec() const override {
        return mps_ctx_.round_prec();
    }

    double round(double x) const override;

    double round(int64_t m, exp_t exp) const override;
};

} // namespace fpy
