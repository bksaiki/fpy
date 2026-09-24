#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>
#include <tuple>

namespace fpy_models::model_amd_cdna2_f16 {

float flush_subnormal__5902d066__105f73b9__bfb0ebf51(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
}

float flush_subnormal_signed__5902d066__29235c5f__ba2817bb5(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_add__5902d066__03265d80__b93816997(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__5902d066__29235c5f__ba2817bb5(z, 1.1754943508222875e-38);
}

float flush_subnormal__5902d066__1c3061a2__b968948b0(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
}

float flush_subnormal_signed__5902d066__645becd2__ba368b368(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_mul__5902d066__88d81c19__b8cddbde6(float x, float y) {
    float x_1 = flush_subnormal__5902d066__1c3061a2__b968948b0(x, 6.103515625e-05);
    float y_1 = flush_subnormal__5902d066__1c3061a2__b968948b0(y, 6.103515625e-05);
    float _t = (x_1 * y_1);
    return flush_subnormal_signed__5902d066__645becd2__ba368b368(_t, 1.1754943508222875e-38);
}

float flush_subnormal_signed__5902d066__c3a63db9__ba19c6238(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_add__5902d066__c4b9d3cb__b0b9f662e(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__5902d066__c3a63db9__ba19c6238(z, 1.1754943508222875e-38);
}

float flush_subnormal_signed__5902d066__ccf0249f__b96cc3044(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_add__5902d066__2afc8375__b6fdb8cbe(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__5902d066__ccf0249f__b96cc3044(z, 1.1754943508222875e-38);
}

float ftz_block__5902d066__f17da570__b0abace67(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    float s = ftz_add__5902d066__03265d80__b93816997(ftz_mul__5902d066__88d81c19__b8cddbde6(A[static_cast<size_t>(0)], B[static_cast<size_t>(0)]), ftz_mul__5902d066__88d81c19__b8cddbde6(A[static_cast<size_t>(1)], B[static_cast<size_t>(1)]));
    float s2 = ftz_add__5902d066__03265d80__b93816997(ftz_mul__5902d066__88d81c19__b8cddbde6(A[static_cast<size_t>(2)], B[static_cast<size_t>(2)]), ftz_mul__5902d066__88d81c19__b8cddbde6(A[static_cast<size_t>(3)], B[static_cast<size_t>(3)]));
    float s_1 = ftz_add__5902d066__c4b9d3cb__b0b9f662e(s, s2);
    return ftz_add__5902d066__2afc8375__b6fdb8cbe(c, s_1);
}

float ftz_addmul(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    float d = flush_subnormal__5902d066__105f73b9__bfb0ebf51(c, 1.1754943508222875e-38);
    for (int8_t i = 0; i < 4; i += 4) {
        std::array<float, 4> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 4)), _tmp1.begin());
        std::array<float, 4> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 4)), _tmp2.begin());
        d = ftz_block__5902d066__f17da570__b0abace67(_tmp1, _tmp2, d);
    }
    return d;
}
}
