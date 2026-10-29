#pragma once

// Run generated fixed-width models over contiguous CPU tensors: one model call
// per row, or, for complete dot products, a chain of calls over each row.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace fpy_models::runners {

using torch::stable::Tensor;

template <typename T, std::size_t N>
const std::array<T, N>& unsafe_array_view(const T* data) {
  // The generated models accept std::array references, while a contiguous
  // tensor already stores each batch row as N adjacent T objects. Avoid a
  // copy by relying on std::array's implementation layout. This is
  // intentionally unsafe: the tensor storage does not formally contain a
  // std::array object under the C++ object-lifetime rules.
  static_assert(std::is_standard_layout_v<std::array<T, N>>);
  static_assert(sizeof(std::array<T, N>) == N * sizeof(T));
  static_assert(alignof(std::array<T, N>) <= alignof(T));
  return *reinterpret_cast<const std::array<T, N>*>(data);
}

template <typename Compute, std::size_t K, auto Model, bool Chain = false>
Tensor run_dpa(const Tensor& a, const Tensor& b, double c,
               torch::headeronly::ScalarType compute_type) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()),
                  "a and b must have the same shape");
  STD_TORCH_CHECK(a.dim() >= 1, "a and b must have at least one dimension");
  STD_TORCH_CHECK((Chain ? (a.sizes().back() > 0 && a.sizes().back() % K == 0)
                         : a.sizes().back() == static_cast<int64_t>(K)),
                  "the last dimension must have size (or, for dot, be a "
                  "positive multiple of) ",
                  K);
  STD_TORCH_CHECK(a.scalar_type() == compute_type, "a has the wrong dtype");
  STD_TORCH_CHECK(b.scalar_type() == compute_type, "b has the wrong dtype");
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU,
                  "a must be a CPU tensor");
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU,
                  "b must be a CPU tensor");

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  std::vector<int64_t> output_shape(a.sizes().begin(), a.sizes().end() - 1);
  Tensor output =
      torch::stable::new_empty(a_contig, output_shape, compute_type);

  const Compute* a_ptr = a_contig.const_data_ptr<Compute>();
  const Compute* b_ptr = b_contig.const_data_ptr<Compute>();
  Compute* output_ptr = output.mutable_data_ptr<Compute>();
  const int64_t length = a.sizes().back();
  const int64_t batch_size = a.numel() / length;

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    Compute acc = static_cast<Compute>(c);
    for (int64_t chunk = 0; chunk < length; chunk += K) {
      const auto offset = batch * length + chunk;
      const auto& a_values = unsafe_array_view<Compute, K>(a_ptr + offset);
      const auto& b_values = unsafe_array_view<Compute, K>(b_ptr + offset);
      acc = Model(a_values, b_values, acc);
    }
    output_ptr[batch] = acc;
  }

  return output;
}

template <typename Compute, std::size_t K, auto Model>
Tensor run_scaled_dpa(const Tensor& a, const Tensor& b, double c, double alpha,
                      double beta, torch::headeronly::ScalarType compute_type) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()),
                  "a and b must have the same shape");
  STD_TORCH_CHECK(a.dim() >= 1, "a and b must have at least one dimension");
  STD_TORCH_CHECK(a.sizes().back() == static_cast<int64_t>(K),
                  "the last dimension must have size ", K);
  STD_TORCH_CHECK(a.scalar_type() == compute_type, "a has the wrong dtype");
  STD_TORCH_CHECK(b.scalar_type() == compute_type, "b has the wrong dtype");
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU,
                  "a must be a CPU tensor");
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU,
                  "b must be a CPU tensor");

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  std::vector<int64_t> output_shape(a.sizes().begin(), a.sizes().end() - 1);
  Tensor output =
      torch::stable::new_empty(a_contig, output_shape, compute_type);

  const Compute* a_ptr = a_contig.const_data_ptr<Compute>();
  const Compute* b_ptr = b_contig.const_data_ptr<Compute>();
  Compute* output_ptr = output.mutable_data_ptr<Compute>();
  const int64_t length = a.sizes().back();
  const int64_t batch_size = a.numel() / length;

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    const auto offset = batch * length;
    const auto& a_values = unsafe_array_view<Compute, K>(a_ptr + offset);
    const auto& b_values = unsafe_array_view<Compute, K>(b_ptr + offset);
    output_ptr[batch] =
        Model(a_values, b_values, static_cast<Compute>(c),
              static_cast<Compute>(alpha), static_cast<Compute>(beta));
  }

  return output;
}

template <typename Compute, std::size_t K, std::size_t ScaleK, auto Model,
          bool Chain = false>
Tensor run_group_scaled_dpa(const Tensor& a, const Tensor& b, double c,
                            const Tensor& alphas, const Tensor& betas,
                            torch::headeronly::ScalarType compute_type) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()),
                  "a and b must have the same shape");
  STD_TORCH_CHECK(a.dim() >= 1, "a and b must have at least one dimension");
  STD_TORCH_CHECK((Chain ? (a.sizes().back() > 0 && a.sizes().back() % K == 0)
                         : a.sizes().back() == static_cast<int64_t>(K)),
                  "the last dimension of a and b must have size (or, for dot, "
                  "be a positive multiple of) ",
                  K);
  STD_TORCH_CHECK(alphas.sizes().equals(betas.sizes()),
                  "alphas and betas must have the same shape");
  STD_TORCH_CHECK(
      alphas.dim() == a.dim(),
      "scale tensors must have the same number of dimensions as a and b");
  const int64_t scales_per_row =
      a.sizes().back() / static_cast<int64_t>(K) * ScaleK;
  STD_TORCH_CHECK(alphas.sizes().back() == scales_per_row,
                  "the last dimension of alphas and betas must have size ",
                  scales_per_row, " (", ScaleK, " per ", K, " elements)");
  for (int64_t dim = 0; dim < a.dim() - 1; ++dim) {
    STD_TORCH_CHECK(
        alphas.sizes()[dim] == a.sizes()[dim],
        "scale tensors must have the same batch dimensions as a and b");
  }
  STD_TORCH_CHECK(a.scalar_type() == compute_type, "a has the wrong dtype");
  STD_TORCH_CHECK(b.scalar_type() == compute_type, "b has the wrong dtype");
  STD_TORCH_CHECK(alphas.scalar_type() == compute_type,
                  "alphas has the wrong dtype");
  STD_TORCH_CHECK(betas.scalar_type() == compute_type,
                  "betas has the wrong dtype");
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU,
                  "a must be a CPU tensor");
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU,
                  "b must be a CPU tensor");
  STD_TORCH_CHECK(alphas.device().type() == torch::headeronly::DeviceType::CPU,
                  "alphas must be a CPU tensor");
  STD_TORCH_CHECK(betas.device().type() == torch::headeronly::DeviceType::CPU,
                  "betas must be a CPU tensor");

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  Tensor alphas_contig = torch::stable::contiguous(alphas);
  Tensor betas_contig = torch::stable::contiguous(betas);
  std::vector<int64_t> output_shape(a.sizes().begin(), a.sizes().end() - 1);
  Tensor output =
      torch::stable::new_empty(a_contig, output_shape, compute_type);

  const Compute* a_ptr = a_contig.const_data_ptr<Compute>();
  const Compute* b_ptr = b_contig.const_data_ptr<Compute>();
  const Compute* alphas_ptr = alphas_contig.const_data_ptr<Compute>();
  const Compute* betas_ptr = betas_contig.const_data_ptr<Compute>();
  Compute* output_ptr = output.mutable_data_ptr<Compute>();
  const int64_t length = a.sizes().back();
  const int64_t batch_size = a.numel() / length;

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    Compute acc = static_cast<Compute>(c);
    for (int64_t chunk = 0; chunk < length; chunk += K) {
      const auto offset = batch * length + chunk;
      const auto scale_offset =
          batch * scales_per_row + chunk / static_cast<int64_t>(K) * ScaleK;
      const auto& a_values = unsafe_array_view<Compute, K>(a_ptr + offset);
      const auto& b_values = unsafe_array_view<Compute, K>(b_ptr + offset);
      if constexpr (ScaleK == 1) {
        // MX models take each block's scale as a scalar.
        acc = Model(a_values, b_values, acc, alphas_ptr[scale_offset],
                    betas_ptr[scale_offset]);
      } else {
        const auto& alpha_values =
            unsafe_array_view<Compute, ScaleK>(alphas_ptr + scale_offset);
        const auto& beta_values =
            unsafe_array_view<Compute, ScaleK>(betas_ptr + scale_offset);
        acc = Model(a_values, b_values, acc, alpha_values, beta_values);
      }
    }
    output_ptr[batch] = acc;
  }

  return output;
}

}  // namespace fpy_models::runners
