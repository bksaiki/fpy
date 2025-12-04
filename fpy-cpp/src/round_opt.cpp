#include <cmath>
#include <iostream>

#include "fpy/params.hpp"
#include "fpy/round_opt.hpp"

namespace fpy {

namespace round_opt {

double round(double x, prec_t p, std::optional<exp_t> n, RM rm) {
    using FP = ieee754_consts<11, 64>; // double precision

    // Fast path: if precision is full precision, no rounding needed
    if (p >= FP::P) {
        return x;
    }

    // Fast path: special values (infinity, NaN, zero)
    if (!std::isfinite(x) || x == 0.0) {
        return x;
    }

    // load floating-point data as integer
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const bool s = (b >> (FP::N - 1)) != 0;
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // decoded exponent and mantissa
    exp_t e;
    mant_t c;

    // decode floating-point data
    if (ebits == 0) {
        // subnormal
        const auto lz = FP::P - std::bit_width(mbits);
        e = FP::EMIN - lz;
        c = mbits << lz;
    } else {
        // normal (assuming no infinity or NaN)
        e = ebits - FP::BIAS;
        c = FP::IMPLICIT1 | mbits;
    }

    // our precision might be limited by subnormalization
    if (n.has_value()) {
        const exp_t nx = e - p;
        const exp_t offset = *n - nx;
        if (offset > 0) {
            // precision reduced due to subnormalization
            p = (static_cast<prec_t>(offset) >= p) ? 0 : p - static_cast<prec_t>(offset);
        }
    }

    // split off discarded bits
    const prec_t p_lost = FP::P - p;
    const mant_t c_mask = bitmask<mant_t>(p_lost);
    const mant_t c_lost = c & c_mask;

    // fast path: exact result
    if (c_lost == 0) {
        return x;
    }

    // clear discarded bits
    c &= ~c_mask;

    // extract rounding information
    // -1: below halfway
    //  0: exactly halfway
    //  1: above halfway
    const mant_t one = 1ULL << p_lost;
    const mant_t halfway = 1ULL << (p_lost - 1);
    const int8_t rb = static_cast<int8_t>(c_lost > halfway) - static_cast<int8_t>(c_lost < halfway);

    // should we increment?
    // case split on nearest
    bool incrementp;
    if (is_nearest(rm)) {
        // nearest rounding mode
        // case split on rounding bits
        if (rb > 0) {
            // above halfway
            incrementp = true;
        } else if (rb == 0) {
            // exactly halfway
            switch (rm) {
                case RM::RTZ: incrementp = false; break;
                case RM::RAZ:
                case RM::RNA: incrementp = true; break;
                case RM::RTP: incrementp = !s; break;
                case RM::RTN: incrementp = s; break;
                case RM::RNE:
                case RM::RTE: incrementp = (c & one) != 0; break;
                case RM::RTO: incrementp = (c & one) == 0; break;
                default: FPY_UNREACHABLE();
            }
        } else {
            // below halfway
            incrementp = false;
        }
    } else {
        // non-nearest
        // case split on rounding mode
        switch (rm) {
            case RM::RTZ: incrementp = false; break;
            case RM::RAZ:
            case RM::RNA: incrementp = true; break;
            case RM::RTP: incrementp = !s; break;
            case RM::RTN: incrementp = s; break;
            case RM::RNE:
            case RM::RTE: incrementp = (c & one) != 0; break;
            case RM::RTO: incrementp = (c & one) == 0; break;
            default: FPY_UNREACHABLE();
        }
    }

    // apply increment
    // if (increment) {
    //     c += one; // "1" at position p
    //     if (c >= (FP::IMPLICIT1 << 1)) {
    //         // mantissa overflowed, adjust exponent
    //         c >>= 1;
    //         e += 1;
    //     }
    // }

    // apply increment
    const mant_t increment = incrementp ? one : 0;
    c += increment;
    
    // check if we need to carry
    const bool carryp = (c >= (FP::IMPLICIT1 << 1));
    c >>= static_cast<uint8_t>(carryp);
    e += static_cast<exp_t>(carryp);

    // encode exponent and mantissa
    uint64_t ebits2, mbits2;
    if (e < FP::EMIN) {
        // subnormal result
        const exp_t shift = FP::EMIN - e;
        ebits2 = 0;
        mbits2 = c >> shift;
    } else {
        // normal result
        ebits2 = e + FP::BIAS;
        mbits2 = c & FP::MMASK;
    }

    // repack the result
    const uint64_t sbits2 = s ? (1ULL << (FP::N - 1)) : 0;
    const uint64_t b2 = sbits2 | (ebits2 << FP::M) | mbits2;
    return std::bit_cast<double>(b2);
}

} // namespace round_opt

} // namespace fpy
