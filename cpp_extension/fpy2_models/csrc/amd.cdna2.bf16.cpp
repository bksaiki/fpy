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

namespace fpy_models::model_amd_cdna2_bf16 {

float flush_subnormal__5902d066__105f73b9__bfb0ebf51(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
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

float ftz_add__5902d066__76f57279__be7d33aa1(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__5902d066__ccf0249f__b96cc3044(z, 1.1754943508222875e-38);
}

float flush_subnormal__5902d066__ae371859__b11ab4587(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
}

float flush_subnormal_signed__5902d066__a8b9086f__b18f79267(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_mul__5902d066__a5b3f37b__b8af70d20(float x, float y) {
    float x_1 = flush_subnormal__5902d066__ae371859__b11ab4587(x, 1.1754943508222875e-38);
    float y_1 = flush_subnormal__5902d066__ae371859__b11ab4587(y, 1.1754943508222875e-38);
    float z = (x_1 * y_1);
    return flush_subnormal_signed__5902d066__a8b9086f__b18f79267(z, 1.1754943508222875e-38);
}

float ftz_add__5902d066__b14caf05__b771bebfe(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__5902d066__ccf0249f__b96cc3044(z, 1.1754943508222875e-38);
}

float ftz_block__5902d066__a9eebad6__b887f9beb(const std::array<float, 2>& A, const std::array<float, 2>& B, float c) {
    float s = ftz_add__5902d066__76f57279__be7d33aa1(ftz_mul__5902d066__a5b3f37b__b8af70d20(A[static_cast<size_t>(0)], B[static_cast<size_t>(0)]), ftz_mul__5902d066__a5b3f37b__b8af70d20(A[static_cast<size_t>(1)], B[static_cast<size_t>(1)]));
    return ftz_add__5902d066__b14caf05__b771bebfe(c, s);
}

float ftz_addmul(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    float d = flush_subnormal__5902d066__105f73b9__bfb0ebf51(c, 1.1754943508222875e-38);
    for (int8_t i = 0; i < 4; i += 2) {
        std::array<float, 2> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 2)), _tmp1.begin());
        std::array<float, 2> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 2)), _tmp2.begin());
        d = ftz_block__5902d066__a9eebad6__b887f9beb(_tmp1, _tmp2, d);
    }
    return d;
}
}
