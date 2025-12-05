#pragma once

#include "context.hpp"
#include "round_opt.hpp"
#include "types.hpp"

namespace fpy {

class MPContext : public Context {
private:
    prec_t prec_;
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
        return prec_;
    }

    inline double round(double x) const override {
        return round_opt::round(x, prec_, std::nullopt, rm_);
    }
};

} // namespace fpy
