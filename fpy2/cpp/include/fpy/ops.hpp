#pragma once

#include "engine.hpp"

namespace fpy {

/// @brief Computes `x + y` to `p` bits of precision.
double add(double x, double y, prec_t p, RM rm);

} // end namespace fpy
