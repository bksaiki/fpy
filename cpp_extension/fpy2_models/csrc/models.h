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

namespace fpy_models::model_amd_cdna3_bf16 {
float tr_fdpa(
    const std::array<float, 8>&,
    const std::array<float, 8>&,
    float);
}

namespace fpy_models::model_amd_cdna3_bf8 {
float gtr_fdpa(
    const std::array<float, 16>&,
    const std::array<float, 16>&,
    float);
}

namespace fpy_models::model_amd_cdna3_f16 {
float tr_fdpa(
    const std::array<float, 8>&,
    const std::array<float, 8>&,
    float);
}

namespace fpy_models::model_nv_ada_e5m2_f32 {
double t_fdpa_chain(
    const std::array<float, 16>&,
    const std::array<float, 16>&,
    float);
}

namespace fpy_models::model_nv_ampere_bf16_f32 {
double t_fdpa_chain(
    const std::array<float, 16>&,
    const std::array<float, 16>&,
    float);
}

namespace fpy_models::model_nv_ampere_tf32_f32 {
double t_fdpa_chain(
    const std::array<float, 8>&,
    const std::array<float, 8>&,
    float);
}

namespace fpy_models::model_nv_blackwell_mxfp8 {
float st_fdpa(
    const std::array<float, 32>&,
    const std::array<float, 32>&,
    float,
    float,
    float);
}

namespace fpy_models::model_nv_blackwell_nvfp4 {
float gst_fdpa(
    const std::array<float, 64>&,
    const std::array<float, 64>&,
    float,
    const std::array<float, 4>&,
    const std::array<float, 4>&);
}

namespace fpy_models::model_nv_hopper_f16_f32 {
float t_fdpa_chain(
    const std::array<float, 32>&,
    const std::array<float, 32>&,
    float);
}

namespace fpy_models::model_nv_turing_f16_f32 {
float t_fdpa_chain(
    const std::array<float, 16>&,
    const std::array<float, 16>&,
    float);
}

namespace fpy_models::model_nv_volta_f16_f32 {
float t_fdpa_chain(
    const std::array<float, 8>&,
    const std::array<float, 8>&,
    float);
}
