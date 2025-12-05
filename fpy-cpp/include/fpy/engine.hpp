#pragma once

#include "types.hpp"

namespace fpy {

namespace engine {

/// @brief Computes `x + y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double add(double x, double y, prec_t p);

/// @brief Computes `x - y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double sub(double x, double y, prec_t p);

/// @brief Computes `x * y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double mul(double x, double y, prec_t p);

/// @brief Computes `x / y` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double div(double x, double y, prec_t p);

/// @brief Computes `sqrt(x)` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double sqrt(double x, prec_t p);

/// @brief Computes `x * y + z` using round-to-odd arithmetic.
///
/// Ensures the result has at least `p` bits of precision.
/// Otherwise, an exception is thrown.
double fma(double x, double y, double z, prec_t p);

} // end namespace engine

} // end namespace fpy
