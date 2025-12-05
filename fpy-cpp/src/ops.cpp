#include <cmath>

#include "fpy/context.hpp"
#include "fpy/ops.hpp"
#include "fpy/real.hpp"
#include "fpy/round_opt.hpp"

namespace fpy {

double round(double x, const Context& ctx) {
    return ctx.round(x);
}

double neg(double x, const Context& ctx) {
    // negate exactly
    x = -x;

    // use context to round
    return ctx.round(x);
}

double abs(double x, const Context& ctx) {
    // take absolute value exactly
    x = std::abs(x);

    // use context to round
    return ctx.round(x);
}

double add(double x, double y, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::add(x, y, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

double sub(double x, double y, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::sub(x, y, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

double mul(double x, double y, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::mul(x, y, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

double div(double x, double y, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::div(x, y, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

double sqrt(double x, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::sqrt(x, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

double fma(double x, double y, double z, const Context& ctx) {
    // compute result using RTO engine
    const double r = engine::fma(x, y, z, ctx.round_prec());

    // use context to round
    return ctx.round(r);
}

} // end namespace fpy
