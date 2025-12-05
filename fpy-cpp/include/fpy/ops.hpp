#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "round.hpp"

namespace fpy {

/// @brief Rounds `x` according to the given context.
double round(double x, const Context& ctx);

/// @brief Computes `-x` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double neg(double x, const Context& ctx);

/// @brief Computes `|x|` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double abs(double x, const Context& ctx);

/// @brief Computes `x + y` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double add(double x, double y, const Context& ctx);

/// @brief Computes `x - y` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double sub(double x, double y, const Context& ctx);

/// @brief Computes `x * y` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double mul(double x, double y, const Context& ctx);

/// @brief Computes `x / y` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double div(double x, double y, const Context& ctx);

/// @brief Computes `sqrt(x)` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double sqrt(double x, const Context& ctx);

/// @brief Computes `x * y + z` using the given context.
/// Must be the case that `ctx.round_prec() <= 64`.
double fma(double x, double y, double z, const Context& ctx);

} // end namespace fpy
