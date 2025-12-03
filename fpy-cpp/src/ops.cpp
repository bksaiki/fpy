#include <cmath>

#include "fpy/ops.hpp"
#include "fpy/real.hpp"
#include "fpy/round_opt.hpp"

namespace fpy {

double neg(double x, prec_t p, RM rm) {
    // negate exactly
    x = -x;

    // fast path for NaN and infinity
    if (std::isnan(x) || std::isinf(x)) {
        return x;
    }

    // negate and round
    auto result = RealFloat(x).round(p, std::nullopt, rm);
    return static_cast<double>(result);
}

double abs(double x, prec_t p, RM rm) {
    // take absolute value exactly
    x = std::abs(x);

    // fast path for NaN and infinity
    if (std::isnan(x) || std::isinf(x)) {
        return x;
    }

    // take absolute value and round
    auto result = RealFloat(x).round(p, std::nullopt, rm);
    return static_cast<double>(result);
}

double add(double x, double y, prec_t p, RM rm) {
    // compute result using RTO engine
    const double r = engine::add(x, y, p + 2);

    // use rounding library
    return round_opt::round(r, p, std::nullopt, rm);
}

double sub(double x, double y, prec_t p, RM rm) {
    // compute result using RTO engine
    const double r = engine::sub(x, y, p + 2);

    // use rounding library
    return round_opt::round(r, p, std::nullopt, rm);
}

double mul(double x, double y, prec_t p, RM rm) {
    // compute result using RTO engine
    const double r = engine::mul(x, y, p + 2);

    // use rounding library
    return round_opt::round(r, p, std::nullopt, rm);
}

double div(double x, double y, prec_t p, RM rm) {
    // compute result using RTO engine
    const double r = engine::div(x, y, p + 2);

    // use rounding library
    return round_opt::round(r, p, std::nullopt, rm);
}

double sqrt(double x, prec_t p, RM rm) {
    // compute result using RTO engine
    const double r = engine::sqrt(x, p + 2);

    // use rounding library
    return round_opt::round(r, p, std::nullopt, rm);
}

double fma(double x, double y, double z, prec_t p, RM rm) {
    // compute result using RTO engine
    const double r = engine::fma(x, y, z, p + 2);

    // use rounding library
    return round_opt::round(r, p, std::nullopt, rm);
}

} // end namespace fpy
