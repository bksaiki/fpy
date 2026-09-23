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

float flush_subnormal__3b330731__930ccb03(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
}

float flush_subnormal_signed__3b330731__6e48a411(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_add__3b330731__c5c4ac89(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__3b330731__6e48a411(z, 1.1754943508222875e-38);
}

float flush_subnormal__3b330731__9a271f2e(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
}

float flush_subnormal_signed__3b330731__58baffc9(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_mul__3b330731__c04d1d03(float x, float y) {
    float x_1 = flush_subnormal__3b330731__9a271f2e(x, 6.103515625e-05);
    float y_1 = flush_subnormal__3b330731__9a271f2e(y, 6.103515625e-05);
    float _t = (x_1 * y_1);
    return flush_subnormal_signed__3b330731__58baffc9(_t, 1.1754943508222875e-38);
}

float ftz_add__3b330731__de92435c(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__3b330731__6e48a411(z, 1.1754943508222875e-38);
}

float flush_subnormal_signed__3b330731__98b99005(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_add__3b330731__cfe744ed(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__3b330731__98b99005(z, 1.1754943508222875e-38);
}

float ftz_block__3b330731__c95405ad(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    float s = ftz_add__3b330731__c5c4ac89(ftz_mul__3b330731__c04d1d03(A[static_cast<size_t>(0)], B[static_cast<size_t>(0)]), ftz_mul__3b330731__c04d1d03(A[static_cast<size_t>(1)], B[static_cast<size_t>(1)]));
    float s2 = ftz_add__3b330731__c5c4ac89(ftz_mul__3b330731__c04d1d03(A[static_cast<size_t>(2)], B[static_cast<size_t>(2)]), ftz_mul__3b330731__c04d1d03(A[static_cast<size_t>(3)], B[static_cast<size_t>(3)]));
    float s_1 = ftz_add__3b330731__de92435c(s, s2);
    return ftz_add__3b330731__cfe744ed(c, s_1);
}

float ftz_addmul(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    float d = flush_subnormal__3b330731__930ccb03(c, 1.1754943508222875e-38);
    for (int8_t i = 0; i < static_cast<uint8_t>(A.size()); i += 4) {
        std::array<float, 4> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 4)), _tmp1.begin());
        std::array<float, 4> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 4)), _tmp2.begin());
        d = ftz_block__3b330731__c95405ad(_tmp1, _tmp2, d);
    }
    return d;
}
}
