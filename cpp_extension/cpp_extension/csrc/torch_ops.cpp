#include <Python.h>

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

template <typename Input, typename Compute, std::size_t K, auto Model>
Tensor run_dpa(
    const Tensor& a,
    const Tensor& b,
    double c,
    torch::headeronly::ScalarType input_type,
    torch::headeronly::ScalarType output_type) {
  STD_TORCH_CHECK(
      a.sizes().equals(b.sizes()), "a and b must have the same shape");
  STD_TORCH_CHECK(
      a.dim() >= 1, "a and b must have at least one dimension");
  STD_TORCH_CHECK(
      a.sizes().back() == static_cast<int64_t>(K),
      "the last dimension must have size ",
      K);
  STD_TORCH_CHECK(a.scalar_type() == input_type, "a has the wrong dtype");
  STD_TORCH_CHECK(b.scalar_type() == input_type, "b has the wrong dtype");
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
      torch::stable::new_empty(a_contig, output_shape, output_type);

  const Input* a_ptr = a_contig.const_data_ptr<Input>();
  const Input* b_ptr = b_contig.const_data_ptr<Input>();
  Compute* output_ptr = output.mutable_data_ptr<Compute>();
  const int64_t batch_size = a.numel() / static_cast<int64_t>(K);

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    std::array<Compute, K> a_values{};
    std::array<Compute, K> b_values{};
    for (std::size_t i = 0; i < K; ++i) {
      const auto offset =
          batch * static_cast<int64_t>(K) + static_cast<int64_t>(i);
      a_values[i] = static_cast<Compute>(a_ptr[offset]);
      b_values[i] = static_cast<Compute>(b_ptr[offset]);
    }
    output_ptr[batch] =
        Model(a_values, b_values, static_cast<Compute>(c));
  }

  return output;
}

Tensor fp64_fma_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      double,
      double,
      4,
      &fpy_models::model_fp64_fma::fma_dpa>(
      a,
      b,
      c,
      torch::headeronly::ScalarType::Double,
      torch::headeronly::ScalarType::Double);
}

Tensor amd_cdna2_bf16_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      c10::BFloat16,
      float,
      4,
      &fpy_models::model_amd_cdna2_bf16::ftz_addmul>(
      a,
      b,
      c,
      torch::headeronly::ScalarType::BFloat16,
      torch::headeronly::ScalarType::Float);
}

Tensor amd_cdna2_f16_cpu(const Tensor& a, const Tensor& b, double c) {
  return run_dpa<
      c10::Half,
      float,
      4,
      &fpy_models::model_amd_cdna2_f16::ftz_addmul>(
      a,
      b,
      c,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::Float);
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
}

STABLE_TORCH_LIBRARY_IMPL(fpy2_models, CPU, m) {
  m.impl("fp64_fma", TORCH_BOX(&fp64_fma_cpu));
  m.impl("amd_cdna2_bf16", TORCH_BOX(&amd_cdna2_bf16_cpu));
  m.impl("amd_cdna2_f16", TORCH_BOX(&amd_cdna2_f16_cpu));
}
