#pragma once

#include <cmath>

#include "context.hpp"
#include "engine.hpp"
#include "round.hpp"

namespace fpy {

/// @brief Engine types for arithmetic operations
enum class EngineType {
    RTO,    // Round-to-odd engine  
    EXACT,  // Exact computation engine
    FIXED   // Fixed-point arithmetic engine
};

/// @brief Rounds `x` according to the given context.
double round(double x, const Context& ctx);

/// @brief Computes `-x` using the given context.
double neg(double x, const Context& ctx);

/// @brief Computes `|x|` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
double abs(double x, const Context& ctx);

/// @brief Computes `x + y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<EngineType E = EngineType::RTO>
double add(double x, double y, const Context& ctx);

/// @brief Computes `x - y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<EngineType E = EngineType::RTO>
double sub(double x, double y, const Context& ctx);

/// @brief Computes `x * y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
template<EngineType E = EngineType::RTO>
double mul(double x, double y, const Context& ctx);

/// @brief Computes `x / y` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
double div(double x, double y, const Context& ctx);

/// @brief Computes `sqrt(x)` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
double sqrt(double x, const Context& ctx);

/// @brief Computes `x * y + z` using the given context.
/// Must be the case that `ctx.round_prec() <= 53`.
double fma(double x, double y, double z, const Context& ctx);

// Template implementations

template<EngineType E>
double add(double x, double y, const Context& ctx) {
    if constexpr (E == EngineType::RTO) {
        // compute result using RTO engine
        const double r = engine::add(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::EXACT) {
        // compute result using exact engine
        const double r = engine::add_exact(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    }
}

template<EngineType E>
double sub(double x, double y, const Context& ctx) {
    if constexpr (E == EngineType::RTO) {
        // compute result using RTO engine
        const double r = engine::sub(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::EXACT) {
        // compute result using exact engine
        const double r = engine::sub_exact(x, y, ctx.round_prec());
        // use context to round
        return ctx.round(r);
    }
}

template<EngineType E>
double mul(double x, double y, const Context& ctx) {
    const prec_t p = ctx.round_prec();
    if constexpr (E == EngineType::RTO) {
        // compute result using RTO engine
        const double r = engine::mul(x, y, p);
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::EXACT) {
        // compute result using exact engine
        const double r = engine::mul_exact(x, y, p);
        // use context to round
        return ctx.round(r);
    } else if constexpr (E == EngineType::FIXED) {
        // compute result using fixed-point arithmetic engine
        if (std::isfinite(x) && std::isfinite(y)) {
            // we can use fixed-point arithmetic
            auto [m, exp] = engine::mul_fixed(x, y, p);
            // round the fixed-point result
            return ctx.round(m, exp);
        } else {
            // special value so use exact engine
            const double r = engine::mul_exact(x, y, p);
            // use context to round
            return ctx.round(r);
        }
    }
}

} // end namespace fpy
