#include <cmath>

#include "fpy/params.hpp"
#include "fpy/round_opt.hpp"

namespace fpy {

namespace round_opt {

double round(double x, prec_t p, const std::optional<exp_t>& n, RM rm) {
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

    // decode floating-point data
    exp_t e;
    mant_t c;
    if (UNLIKELY(ebits == 0)) {
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
    bool overshiftp = false;
    if (n.has_value()) {
        const exp_t nx = e - p;
        const exp_t offset = *n - nx;
        if (offset > 0) {
            // precision reduced due to subnormalization
            // "overshift" is set if we shift more than p bits
            const prec_t offset_pos = static_cast<prec_t>(offset);
            bool overshiftp = offset_pos > p; // set overshift flag
            p = overshiftp ? 0 : p - offset_pos; // precision cannot be negative
            e = overshiftp ? *n : e; // overshift implies e < n, set for correct increment to MIN_VAL
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

    // value of the LSB for precision p
    const mant_t one = 1ULL << p_lost;

    // should we increment?
    // case split on nearest
    bool incrementp;
    if (is_nearest(rm)) {
        // nearest rounding

        // clever way to extract rounding information
        // -1: below halfway
        //  0: exactly halfway
        //  1: above halfway
        const mant_t halfway = 1ULL << (p_lost - 1);
        const int8_t cmp = static_cast<int8_t>(c_lost > halfway) - static_cast<int8_t>(c_lost < halfway);
        const int8_t rb = overshiftp ? -1 : cmp; // overshift implies below halfway

        // case split on rounding bits
        if (rb == 0) {
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
            // above or below halfway
            incrementp = rb > 0;
        }
    } else {
        // non-nearest
        // case split on rounding mode
        switch (rm) {
            case RM::RTZ: incrementp = false; break;
            case RM::RAZ: incrementp = true; break;
            case RM::RTP: incrementp = !s; break;
            case RM::RTN: incrementp = s; break;
            case RM::RTE: incrementp = (c & one) != 0; break;
            case RM::RTO: incrementp = (c & one) == 0; break;
            default: FPY_UNREACHABLE();
        }
    }

    // apply increment
    const mant_t increment = incrementp ? one : static_cast<mant_t>(0);
    c += increment;

    // check if we need to carry
    const bool carryp = c >= (FP::IMPLICIT1 << 1);
    e += static_cast<exp_t>(carryp);
    c >>= static_cast<uint8_t>(carryp);

    // encode exponent and mantissa
    uint64_t ebits2, mbits2;
    if (UNLIKELY(c == 0)) {
        // edge case: subnormalization underflowed to 0
        // `e` might be an unexpected value here
        ebits2 = 0;
        mbits2 = 0;
    } else if (UNLIKELY(e < FP::EMIN)) {
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
    const uint64_t sbits2 = static_cast<uint64_t>(s) << (FP::N - 1);
    const uint64_t b2 = sbits2 | (ebits2 << FP::M) | mbits2;
    return std::bit_cast<double>(b2);
}

} // namespace round_opt

} // namespace fpy
