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

namespace fpy_models::model_fp64_fma {

double fma_dpa(const std::array<double, 4>& A, const std::array<double, 4>& B, double c) {
    double d = c;
    for (int8_t _i = 0; _i < 4; ++_i) {
        double a = A[static_cast<size_t>(_i)];
        double b = B[static_cast<size_t>(_i)];
        d = std::fma(a, b, d);
    }
    return d;
}
}
