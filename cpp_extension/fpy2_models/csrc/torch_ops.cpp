#include <Python.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "models.h"

namespace {

using torch::stable::Tensor;

template <typename Compute, std::size_t K, auto Model>
Tensor run_dpa(
    const Tensor& a,
    const Tensor& b,
    double c,
    torch::headeronly::ScalarType compute_type) {
  STD_TORCH_CHECK(
      a.sizes().equals(b.sizes()), "a and b must have the same shape");
  STD_TORCH_CHECK(
      a.dim() >= 1, "a and b must have at least one dimension");
  STD_TORCH_CHECK(
      a.sizes().back() == static_cast<int64_t>(K),
      "the last dimension must have size ",
      K);
  STD_TORCH_CHECK(a.scalar_type() == compute_type, "a has the wrong dtype");
  STD_TORCH_CHECK(b.scalar_type() == compute_type, "b has the wrong dtype");
  STD_TORCH_CHECK(
      a.device().type() == torch::headeronly::DeviceType::CPU,
      "a must be a CPU tensor");
  STD_TORCH_CHECK(
      b.device().type() == torch::headeronly::DeviceType::CPU,
      "b must be a CPU tensor");

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  std::vector<int64_t> output_shape(
      a.sizes().begin(), a.sizes().end() - 1);
  Tensor output =
      torch::stable::new_empty(a_contig, output_shape, compute_type);

  const Compute* a_ptr = a_contig.const_data_ptr<Compute>();
  const Compute* b_ptr = b_contig.const_data_ptr<Compute>();
  Compute* output_ptr = output.mutable_data_ptr<Compute>();
  const int64_t batch_size = a.numel() / static_cast<int64_t>(K);

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    std::array<Compute, K> a_values{};
    std::array<Compute, K> b_values{};
    const auto offset = batch * static_cast<int64_t>(K);
    std::copy_n(a_ptr + offset, K, a_values.begin());
    std::copy_n(b_ptr + offset, K, b_values.begin());
    output_ptr[batch] =
        Model(a_values, b_values, static_cast<Compute>(c));
  }

  return output;
}

template <typename Compute, std::size_t K, auto Model>
Tensor run_scaled_dpa(
    const Tensor& a,
    const Tensor& b,
    double c,
    double alpha,
    double beta,
    torch::headeronly::ScalarType compute_type) {
  STD_TORCH_CHECK(
      a.sizes().equals(b.sizes()), "a and b must have the same shape");
  STD_TORCH_CHECK(
      a.dim() >= 1, "a and b must have at least one dimension");
  STD_TORCH_CHECK(
      a.sizes().back() == static_cast<int64_t>(K),
      "the last dimension must have size ",
      K);
  STD_TORCH_CHECK(a.scalar_type() == compute_type, "a has the wrong dtype");
  STD_TORCH_CHECK(b.scalar_type() == compute_type, "b has the wrong dtype");
  STD_TORCH_CHECK(
      a.device().type() == torch::headeronly::DeviceType::CPU,
      "a must be a CPU tensor");
  STD_TORCH_CHECK(
      b.device().type() == torch::headeronly::DeviceType::CPU,
      "b must be a CPU tensor");

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  std::vector<int64_t> output_shape(
      a.sizes().begin(), a.sizes().end() - 1);
  Tensor output =
      torch::stable::new_empty(a_contig, output_shape, compute_type);

  const Compute* a_ptr = a_contig.const_data_ptr<Compute>();
  const Compute* b_ptr = b_contig.const_data_ptr<Compute>();
  Compute* output_ptr = output.mutable_data_ptr<Compute>();
  const int64_t batch_size = a.numel() / static_cast<int64_t>(K);

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    std::array<Compute, K> a_values{};
    std::array<Compute, K> b_values{};
    const auto offset = batch * static_cast<int64_t>(K);
    std::copy_n(a_ptr + offset, K, a_values.begin());
    std::copy_n(b_ptr + offset, K, b_values.begin());
    output_ptr[batch] = Model(
        a_values,
        b_values,
        static_cast<Compute>(c),
        static_cast<Compute>(alpha),
        static_cast<Compute>(beta));
  }

  return output;
}

template <
    typename Compute,
    std::size_t K,
    std::size_t ScaleK,
    auto Model>
Tensor run_group_scaled_dpa(
    const Tensor& a,
    const Tensor& b,
    double c,
    const Tensor& alphas,
    const Tensor& betas,
    torch::headeronly::ScalarType compute_type) {
  STD_TORCH_CHECK(
      a.sizes().equals(b.sizes()), "a and b must have the same shape");
  STD_TORCH_CHECK(
      a.dim() >= 1, "a and b must have at least one dimension");
  STD_TORCH_CHECK(
      a.sizes().back() == static_cast<int64_t>(K),
      "the last dimension of a and b must have size ",
      K);
  STD_TORCH_CHECK(
      alphas.sizes().equals(betas.sizes()),
      "alphas and betas must have the same shape");
  STD_TORCH_CHECK(
      alphas.dim() == a.dim(),
      "scale tensors must have the same number of dimensions as a and b");
  STD_TORCH_CHECK(
      alphas.sizes().back() == static_cast<int64_t>(ScaleK),
      "the last dimension of alphas and betas must have size ",
      ScaleK);
  for (int64_t dim = 0; dim < a.dim() - 1; ++dim) {
    STD_TORCH_CHECK(
        alphas.sizes()[dim] == a.sizes()[dim],
        "scale tensors must have the same batch dimensions as a and b");
  }
  STD_TORCH_CHECK(a.scalar_type() == compute_type, "a has the wrong dtype");
  STD_TORCH_CHECK(b.scalar_type() == compute_type, "b has the wrong dtype");
  STD_TORCH_CHECK(
      alphas.scalar_type() == compute_type, "alphas has the wrong dtype");
  STD_TORCH_CHECK(
      betas.scalar_type() == compute_type, "betas has the wrong dtype");
  STD_TORCH_CHECK(
      a.device().type() == torch::headeronly::DeviceType::CPU,
      "a must be a CPU tensor");
  STD_TORCH_CHECK(
      b.device().type() == torch::headeronly::DeviceType::CPU,
      "b must be a CPU tensor");
  STD_TORCH_CHECK(
      alphas.device().type() == torch::headeronly::DeviceType::CPU,
      "alphas must be a CPU tensor");
  STD_TORCH_CHECK(
      betas.device().type() == torch::headeronly::DeviceType::CPU,
      "betas must be a CPU tensor");

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  Tensor alphas_contig = torch::stable::contiguous(alphas);
  Tensor betas_contig = torch::stable::contiguous(betas);
  std::vector<int64_t> output_shape(
      a.sizes().begin(), a.sizes().end() - 1);
  Tensor output =
      torch::stable::new_empty(a_contig, output_shape, compute_type);

  const Compute* a_ptr = a_contig.const_data_ptr<Compute>();
  const Compute* b_ptr = b_contig.const_data_ptr<Compute>();
  const Compute* alphas_ptr = alphas_contig.const_data_ptr<Compute>();
  const Compute* betas_ptr = betas_contig.const_data_ptr<Compute>();
  Compute* output_ptr = output.mutable_data_ptr<Compute>();
  const int64_t batch_size = a.numel() / static_cast<int64_t>(K);

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    std::array<Compute, K> a_values{};
    std::array<Compute, K> b_values{};
    std::array<Compute, ScaleK> alpha_values{};
    std::array<Compute, ScaleK> beta_values{};
    const auto offset = batch * static_cast<int64_t>(K);
    const auto scale_offset = batch * static_cast<int64_t>(ScaleK);
    std::copy_n(a_ptr + offset, K, a_values.begin());
    std::copy_n(b_ptr + offset, K, b_values.begin());
    std::copy_n(alphas_ptr + scale_offset, ScaleK, alpha_values.begin());
    std::copy_n(betas_ptr + scale_offset, ScaleK, beta_values.begin());
    output_ptr[batch] = Model(
        a_values,
        b_values,
        static_cast<Compute>(c),
        alpha_values,
        beta_values);
  }

  return output;
}

Tensor fp64_fma_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      double,
      4,
      &fpy_models::model_fp64_fma::fma_dpa>(
      a,
      b,
      c,
      torch::headeronly::ScalarType::Double);
}

Tensor amd_cdna2_bf16_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      4,
      &fpy_models::model_amd_cdna2_bf16::ftz_addmul>(
      a,
      b,
      c,
      torch::headeronly::ScalarType::Float);
}

Tensor amd_cdna2_f16_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      4,
      &fpy_models::model_amd_cdna2_f16::ftz_addmul>(
      a,
      b,
      c,
      torch::headeronly::ScalarType::Float);
}

Tensor amd_cdna3_bf16_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      8,
      &fpy_models::model_amd_cdna3_bf16::tr_fdpa>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor amd_cdna3_bf8_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      16,
      &fpy_models::model_amd_cdna3_bf8::gtr_fdpa>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor amd_cdna3_f16_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      8,
      &fpy_models::model_amd_cdna3_f16::tr_fdpa>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor nv_ada_e5m2_f32_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      16,
      &fpy_models::model_nv_ada_e5m2_f32::t_fdpa_chain>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor nv_ampere_bf16_f32_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      16,
      &fpy_models::model_nv_ampere_bf16_f32::t_fdpa_chain>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor nv_ampere_tf32_f32_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      8,
      &fpy_models::model_nv_ampere_tf32_f32::t_fdpa_chain>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor nv_blackwell_mxfp8_cpu(
    const Tensor& a,
    const Tensor& b,
    double c,
    double alpha,
    double beta) {
  return run_scaled_dpa<
      float,
      32,
      &fpy_models::model_nv_blackwell_mxfp8::st_fdpa>(
      a, b, c, alpha, beta, torch::headeronly::ScalarType::Float);
}

Tensor nv_blackwell_nvfp4_cpu(
    const Tensor& a,
    const Tensor& b,
    double c,
    const Tensor& alphas,
    const Tensor& betas) {
  return run_group_scaled_dpa<
      float,
      64,
      4,
      &fpy_models::model_nv_blackwell_nvfp4::gst_fdpa>(
      a,
      b,
      c,
      alphas,
      betas,
      torch::headeronly::ScalarType::Float);
}

Tensor nv_hopper_f16_f32_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      32,
      &fpy_models::model_nv_hopper_f16_f32::t_fdpa_chain>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor nv_turing_f16_f32_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      16,
      &fpy_models::model_nv_turing_f16_f32::t_fdpa_chain>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

Tensor nv_volta_f16_f32_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      float,
      8,
      &fpy_models::model_nv_volta_f16_f32::t_fdpa_chain>(
      a, b, c, torch::headeronly::ScalarType::Float);
}

}  // namespace

extern "C" PyObject* PyInit__C(void) {
  static PyModuleDef module = {
      PyModuleDef_HEAD_INIT,
      "_C",
      nullptr,
      -1,
      nullptr,
  };
  return PyModule_Create(&module);
}

STABLE_TORCH_LIBRARY(fpy2_models, m) {
  m.def("fp64_fma(Tensor a, Tensor b, float c) -> Tensor");
  m.def("amd_cdna2_bf16(Tensor a, Tensor b, float c) -> Tensor");
  m.def("amd_cdna2_f16(Tensor a, Tensor b, float c) -> Tensor");
  m.def("amd_cdna3_bf16(Tensor a, Tensor b, float c) -> Tensor");
  m.def("amd_cdna3_bf8(Tensor a, Tensor b, float c) -> Tensor");
  m.def("amd_cdna3_f16(Tensor a, Tensor b, float c) -> Tensor");
  m.def("nv_ada_e5m2_f32(Tensor a, Tensor b, float c) -> Tensor");
  m.def("nv_ampere_bf16_f32(Tensor a, Tensor b, float c) -> Tensor");
  m.def("nv_ampere_tf32_f32(Tensor a, Tensor b, float c) -> Tensor");
  m.def(
      "nv_blackwell_mxfp8(Tensor a, Tensor b, float c, float alpha, "
      "float beta) -> Tensor");
  m.def(
      "nv_blackwell_nvfp4(Tensor a, Tensor b, float c, Tensor alphas, "
      "Tensor betas) -> Tensor");
  m.def("nv_hopper_f16_f32(Tensor a, Tensor b, float c) -> Tensor");
  m.def("nv_turing_f16_f32(Tensor a, Tensor b, float c) -> Tensor");
  m.def("nv_volta_f16_f32(Tensor a, Tensor b, float c) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(fpy2_models, CPU, m) {
  m.impl("fp64_fma", TORCH_BOX(&fp64_fma_cpu));
  m.impl("amd_cdna2_bf16", TORCH_BOX(&amd_cdna2_bf16_cpu));
  m.impl("amd_cdna2_f16", TORCH_BOX(&amd_cdna2_f16_cpu));
  m.impl("amd_cdna3_bf16", TORCH_BOX(&amd_cdna3_bf16_cpu));
  m.impl("amd_cdna3_bf8", TORCH_BOX(&amd_cdna3_bf8_cpu));
  m.impl("amd_cdna3_f16", TORCH_BOX(&amd_cdna3_f16_cpu));
  m.impl("nv_ada_e5m2_f32", TORCH_BOX(&nv_ada_e5m2_f32_cpu));
  m.impl("nv_ampere_bf16_f32", TORCH_BOX(&nv_ampere_bf16_f32_cpu));
  m.impl("nv_ampere_tf32_f32", TORCH_BOX(&nv_ampere_tf32_f32_cpu));
  m.impl("nv_blackwell_mxfp8", TORCH_BOX(&nv_blackwell_mxfp8_cpu));
  m.impl("nv_blackwell_nvfp4", TORCH_BOX(&nv_blackwell_nvfp4_cpu));
  m.impl("nv_hopper_f16_f32", TORCH_BOX(&nv_hopper_f16_f32_cpu));
  m.impl("nv_turing_f16_f32", TORCH_BOX(&nv_turing_f16_f32_cpu));
  m.impl("nv_volta_f16_f32", TORCH_BOX(&nv_volta_f16_f32_cpu));
}
