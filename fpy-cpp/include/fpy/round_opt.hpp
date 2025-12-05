#pragma once

#include <optional>

#include "round.hpp"
#include "types.hpp"

namespace fpy {

namespace round_opt {

/// @brief Optimized rounding to finalize a round-to-odd floating-point result.
/// Assumes that the argument has at least p + 2 bits of precision,
/// where p is the target precision.
double round(double x, prec_t p, const std::optional<exp_t>& n, RM mode);

} // namespace round_opt

} // namespace fpy
