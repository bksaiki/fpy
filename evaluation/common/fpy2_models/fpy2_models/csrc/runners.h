#pragma once

// Run generated fixed-width models over contiguous CPU tensors, with the
// interface of MMA-Sim's `dpa`: operands a and b of shape (..., K), one
// accumulator per row in c, of shape (...), and, for block-scaled models,
// scales alphas and betas of shape (..., S). Each returns the new
// accumulators, shaped like c.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

// Check that t has c's shape followed by width, so that every row can be read
// as a std::array. Torch already checks the rest: typed data_ptr calls check
// the dtype, and the operators are only registered for CPU.
inline void check_rows(const Tensor& t, const char* name, const Tensor& c,
                       int64_t width) {
  STD_TORCH_CHECK(t.dim() == c.dim() + 1 && t.sizes().back() == width, name,
                  " must have shape (*c.shape, ", width, ")");
  for (int64_t dim = 0; dim < c.dim(); ++dim) {
    STD_TORCH_CHECK(t.sizes()[dim] == c.sizes()[dim], name,
                    " must have shape (*c.shape, ", width, ")");
  }
}

template <typename T, std::size_t K, auto Model>
Tensor run_dpa(const Tensor& a, const Tensor& b, const Tensor& c) {
  check_rows(a, "a", c, K);
  check_rows(b, "b", c, K);

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  Tensor c_contig = torch::stable::contiguous(c);
  Tensor output = torch::stable::empty_like(c_contig);

  const T* a_ptr = a_contig.const_data_ptr<T>();
  const T* b_ptr = b_contig.const_data_ptr<T>();
  const T* c_ptr = c_contig.const_data_ptr<T>();
  T* output_ptr = output.mutable_data_ptr<T>();

  for (int64_t batch = 0; batch < c.numel(); ++batch) {
    const auto& a_values = unsafe_array_view<T, K>(a_ptr + batch * K);
    const auto& b_values = unsafe_array_view<T, K>(b_ptr + batch * K);
    output_ptr[batch] = Model(a_values, b_values, c_ptr[batch]);
  }

  return output;
}

// ScaleK is 1 for MX models, which take each row's scale as a scalar, and the
// number of scale groups per row otherwise.
template <typename T, std::size_t K, std::size_t ScaleK, auto Model>
Tensor run_scaled_dpa(const Tensor& a, const Tensor& b, const Tensor& c,
                      const Tensor& alphas, const Tensor& betas) {
  check_rows(a, "a", c, K);
  check_rows(b, "b", c, K);
  check_rows(alphas, "alphas", c, ScaleK);
  check_rows(betas, "betas", c, ScaleK);

  Tensor a_contig = torch::stable::contiguous(a);
  Tensor b_contig = torch::stable::contiguous(b);
  Tensor c_contig = torch::stable::contiguous(c);
  Tensor alphas_contig = torch::stable::contiguous(alphas);
  Tensor betas_contig = torch::stable::contiguous(betas);
  Tensor output = torch::stable::empty_like(c_contig);

  const T* a_ptr = a_contig.const_data_ptr<T>();
  const T* b_ptr = b_contig.const_data_ptr<T>();
  const T* c_ptr = c_contig.const_data_ptr<T>();
  const T* alphas_ptr = alphas_contig.const_data_ptr<T>();
  const T* betas_ptr = betas_contig.const_data_ptr<T>();
  T* output_ptr = output.mutable_data_ptr<T>();

  for (int64_t batch = 0; batch < c.numel(); ++batch) {
    const auto& a_values = unsafe_array_view<T, K>(a_ptr + batch * K);
    const auto& b_values = unsafe_array_view<T, K>(b_ptr + batch * K);
    if constexpr (ScaleK == 1) {
      output_ptr[batch] = Model(a_values, b_values, c_ptr[batch],
                                alphas_ptr[batch], betas_ptr[batch]);
    } else {
      const auto& alpha_values =
          unsafe_array_view<T, ScaleK>(alphas_ptr + batch * ScaleK);
      const auto& beta_values =
          unsafe_array_view<T, ScaleK>(betas_ptr + batch * ScaleK);
      output_ptr[batch] =
          Model(a_values, b_values, c_ptr[batch], alpha_values, beta_values);
    }
  }

  return output;
}

}  // namespace fpy_models::runners
