#include <cmath>

#include "fpy/params.hpp"
#include "fpy/round_opt.hpp"

namespace fpy {

namespace round_opt {

double round(double x, prec_t p, std::optional<exp_t> n, RM rm) {
    using FP = ieee754_consts<11, 64>; // double precision

    // special case: zero
    if (x == 0.0) {
        return x;
    }

    // handle sign separately
    const bool s = std::signbit(x);

    // load floating-point data as integer
    const uint64_t b = std::bit_cast<uint64_t>(x);
    const uint64_t ebits = (b & FP::EMASK) >> FP::M;
    const uint64_t mbits = b & FP::MMASK;

    // decoded exponent and mantissa
    exp_t e;
    mant_t c;

    // decode floating-point data
    if (ebits == 0) {
        // subnormal
        FPY_ASSERT(false, "unimplemented");
    } else if (ebits == FP::EONES) {
        // infinity or NaN
        return x;
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
            p -= static_cast<prec_t>(offset);
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
    const mant_t halfway = one >> 1;
    const int8_t rb = static_cast<int8_t>(c_lost > halfway) - static_cast<int8_t>(c_lost < halfway);

    // should we increment?
    // case split on nearest
    bool increment;
    if (is_nearest(rm)) {
        // nearest rounding mode
        // case split on rounding bits
        switch (rb) {
            case 1:
                // above halfway
                increment = true;
                break;
            case 0:
                // exact halfway
                // case split on rounding mode
                switch (rm) {
                    case RM::RTZ:
                        increment = false;
                        break;

                    case RM::RAZ:
                    case RM::RNA:
                        increment = true;
                        break;

                    case RM::RTP:
                        increment = !s;
                        break;

                    case RM::RTN:
                        increment = s;
                        break;

                    case RM::RNE:
                    case RM::RTE:
                        increment = (c & one);
                        break;

                    case RM::RTO:
                        increment = !(c & one);
                        break;

                    default:
                        FPY_UNREACHABLE();
                }
                break;
            case -1:
                // below halfway
                increment = false;
                break;
            default:
                FPY_UNREACHABLE();
        }
    } else {
        // non-nearest
        // case split on rounding mode
        switch (rm) {
            case RM::RTZ:
                increment = false;
                break;

            case RM::RAZ:
            case RM::RNA:
                increment = true;
                break;

            case RM::RTP:
                increment = !s;
                break;

            case RM::RTN:
                increment = s;
                break;

            case RM::RNE:
            case RM::RTE:
                increment = (c & one);
                break;

            case RM::RTO:
                increment = !(c & one);
                break;

            default:
                FPY_UNREACHABLE();
        }
    }

    // apply increment
    if (increment) {
        c += one; // "1" at position p
        if (c >= (FP::IMPLICIT1 << 1)) {
            // mantissa overflowed, adjust exponent
            c >>= 1;
            e += 1;
        }
    }

    // check if we are subnormal at double-precision
    if (e < FP::EMIN) {
        // subnormal result
        FPY_ASSERT(false, "unimplemented");
    }

    // encoded exponent and mantissa
    const uint64_t sbits2 = s ? (1ULL << (FP::N - 1)) : 0;
    const uint64_t ebits2 = e + FP::BIAS;
    const uint64_t mbits2 = c & FP::MMASK;

    // repack the result
    const uint64_t b2 = sbits2 | (ebits2 << FP::M) | mbits2;
    return std::bit_cast<double>(b2);
}

} // namespace round_opt

} // namespace fpy
