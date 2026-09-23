#pragma once

#include <array>

namespace fpy_models::model_fp64_fma {
double fma_dpa(
    const std::array<double, 4>&,
    const std::array<double, 4>&,
    double);
}

namespace fpy_models::model_amd_cdna2_bf16 {
float ftz_addmul(
    const std::array<float, 4>&,
    const std::array<float, 4>&,
    float);
}

namespace fpy_models::model_amd_cdna2_f16 {
float ftz_addmul(
    const std::array<float, 4>&,
    const std::array<float, 4>&,
    float);
}