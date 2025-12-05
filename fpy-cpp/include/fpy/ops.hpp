#pragma once

#include "engine.hpp"
#include "round.hpp"

namespace fpy {

/// @brief Computes `-x` to `p` bits of precision under
/// rounding mode `rm`.
double neg(double x, prec_t p, RM rm);

/// @brief Computes `|x|` to `p` bits of precision under
/// rounding mode `rm`.
double abs(double x, prec_t p, RM rm);

/// @brief Computes `x + y` to `p` bits of precision under
/// rounding mode `rm`.
double add(double x, double y, prec_t p, RM rm);

/// @brief Computes `x - y` to `p` bits of precision under
/// rounding mode `rm`.
double sub(double x, double y, prec_t p, RM rm);

/// @brief Computes `x * y` to `p` bits of precision under
/// rounding mode `rm`.
double mul(double x, double y, prec_t p, RM rm);

/// @brief Computes `x / y` to `p` bits of precision under
/// rounding mode `rm`.
double div(double x, double y, prec_t p, RM rm);

/// @brief Computes `sqrt(x)` to `p` bits of precision under
/// rounding mode `rm`.
double sqrt(double x, prec_t p, RM rm);

/// @brief Computes `x * y + z` to `p` bits of precision under
/// rounding mode `rm`.
double fma(double x, double y, double z, prec_t p, RM rm);

} // end namespace fpy
