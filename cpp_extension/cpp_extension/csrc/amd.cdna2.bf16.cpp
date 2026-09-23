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

float flush_subnormal__3b330731__930ccb03(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
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

float ftz_add__3b330731__97d62893(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__3b330731__98b99005(z, 1.1754943508222875e-38);
}

float flush_subnormal__3b330731__7d4947b3(float x, float tiny) {
    return ((std::fabs(x) < tiny) ? static_cast<float>(0) : x);
}

float flush_subnormal_signed__3b330731__a8386a36(float x, float tiny) {
    float t{};
    if ((std::fabs(x) < tiny)) {
        t = (x * static_cast<float>(0));
    } else {
        t = x;
    }
    return t;
}

float ftz_mul__3b330731__e5b37d47(float x, float y) {
    float x_1 = flush_subnormal__3b330731__7d4947b3(x, 1.1754943508222875e-38);
    float y_1 = flush_subnormal__3b330731__7d4947b3(y, 1.1754943508222875e-38);
    float z = (x_1 * y_1);
    return flush_subnormal_signed__3b330731__a8386a36(z, 1.1754943508222875e-38);
}

float ftz_add__3b330731__4246420c(float x, float y) {
    float z = (x + y);
    return flush_subnormal_signed__3b330731__98b99005(z, 1.1754943508222875e-38);
}

float ftz_block__3b330731__2afe4926(const std::array<float, 2>& A, const std::array<float, 2>& B, float c) {
    float s = ftz_add__3b330731__97d62893(ftz_mul__3b330731__e5b37d47(A[static_cast<size_t>(0)], B[static_cast<size_t>(0)]), ftz_mul__3b330731__e5b37d47(A[static_cast<size_t>(1)], B[static_cast<size_t>(1)]));
    return ftz_add__3b330731__4246420c(c, s);
}

float ftz_addmul(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    float d = flush_subnormal__3b330731__930ccb03(c, 1.1754943508222875e-38);
    for (int8_t i = 0; i < static_cast<uint8_t>(A.size()); i += 2) {
        std::array<float, 2> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 2)), _tmp1.begin());
        std::array<float, 2> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 2)), _tmp2.begin());
        d = ftz_block__3b330731__2afe4926(_tmp1, _tmp2, d);
    }
    return d;
}
}
