#pragma once

#include "types.hpp"

namespace fpy {

// constexpr version of bitmask
template <typename T, uint64_t k>
constexpr T __bitmask() {
    FPY_STATIC_ASSERT(std::is_integral_v<T> && std::is_unsigned_v<T>, "T must be an unsigned integer");
    FPY_STATIC_ASSERT(k < 64, "exceeded maximum bitmask size");
    FPY_STATIC_ASSERT(k <= 8 * sizeof(T), "exceeded maximum bitmask size for T");
    return static_cast<T>((1ULL << k) - 1);
}

/// @brief Type trait defining IEEE 754 format constants
/// @tparam E bitwidth of the exponent field
/// @tparam N total bitwidth of the format
template <uint64_t E_, uint64_t N_>
struct ieee754_consts {
    static constexpr uint64_t E = E_;
    static constexpr uint64_t N = N_;

    FPY_STATIC_ASSERT(E >= 2, "Invalid IEEE 754 format");
    FPY_STATIC_ASSERT(N >= E + 2, "Invalid IEEE 754 format");
    FPY_STATIC_ASSERT(N <= 64, "Exceeded maximum supported IEEE 754 format");

    static constexpr prec_t P = static_cast<prec_t>(N - E);
    static constexpr prec_t M = P - 1;

    static constexpr exp_t EMAX = static_cast<exp_t>(1ULL << (E - 1)) - 1;
    static constexpr exp_t EMIN = 1 - EMAX;
    static constexpr exp_t EXPMAX = EMAX - static_cast<exp_t>(P) + 1;
    static constexpr exp_t EXPMIN = EMIN - static_cast<exp_t>(P) + 1;
    static constexpr exp_t BIAS = EMAX;

    static constexpr uint64_t SMASK = 1ULL << (N - 1);
    static constexpr uint64_t EMASK = __bitmask<uint64_t, E>() << M;
    static constexpr uint64_t MMASK = __bitmask<uint64_t, M>();

    static constexpr uint64_t EONES = __bitmask<uint64_t, E>();

    static constexpr mant_t IMPLICIT1 = 1ULL << M;
};

} // namespace fpy

