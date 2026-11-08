#pragma once

// Run generated fixed-width models over contiguous CPU tensors, with the
// interface of MMA-Sim's `dpa`: operands a and b of shape (..., K), one
// accumulator per row in c, of shape (...), and, for block-scaled models,
// scales alphas and betas of shape (..., S). Each returns the new
// accumulators, shaped like c.
//
// The GEMM runners chain a model along rows instead: entry p is row I[p] of A
// dotted with row J[p] of Bt, K elements per call, from 0. With every (i, j),
// that is A Bt^T; with I = J = 0..P-1, P paired dot products.

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

// The entries of A Bt^T at (I[p], J[p]): each chains call(i, j, chunk, acc)
// over the chunks of K in a row. The entries split over Torch's intra-op
// threads (torch.set_num_threads); the models restore any rounding mode they
// set, so reusing Torch's threads leaves Torch's own arithmetic as it was.
template <typename T, std::size_t K, typename Call>
Tensor gemm(const Tensor& A, const Tensor& Bt, const Tensor& I, const Tensor& J,
            Call call) {
  STD_TORCH_CHECK(A.dim() == 2 && Bt.dim() == 2 && A.size(1) == Bt.size(1) &&
                      A.size(1) % K == 0,
                  "A and Bt must be matrices with rows of a multiple of ", K);
  STD_TORCH_CHECK(I.dim() == 1 && J.sizes().equals(I.sizes()),
                  "I and J must be vectors of one length");

  Tensor output = torch::stable::new_empty(A, {I.numel()});
  const int64_t* i_ptr = I.const_data_ptr<int64_t>();
  const int64_t* j_ptr = J.const_data_ptr<int64_t>();
  T* output_ptr = output.mutable_data_ptr<T>();
  const int64_t chunks = A.size(1) / K;

  auto range = [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      T acc{};
      for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        acc = call(i_ptr[p], j_ptr[p], chunk, acc);
      }
      output_ptr[p] = acc;
    }
  };

  torch::stable::parallel_for(0, I.numel(), 1, range);
  return output;
}

template <typename T, std::size_t K, auto Model>
Tensor run_gemm(const Tensor& A, const Tensor& Bt, const Tensor& I,
                const Tensor& J) {
  Tensor a = torch::stable::contiguous(A);
  Tensor b = torch::stable::contiguous(Bt);
  const T* a_ptr = a.const_data_ptr<T>();
  const T* b_ptr = b.const_data_ptr<T>();
  const int64_t L = a.size(1);
  return gemm<T, K>(a, b, torch::stable::contiguous(I),
                    torch::stable::contiguous(J),
                    [=](int64_t i, int64_t j, int64_t chunk, T acc) {
                      return Model(
                          unsafe_array_view<T, K>(a_ptr + i * L + chunk * K),
                          unsafe_array_view<T, K>(b_ptr + j * L + chunk * K),
                          acc);
                    });
}

// alphas and betas hold ScaleK scales per chunk of A's and Bt's rows.
template <typename T, std::size_t K, std::size_t ScaleK, auto Model>
Tensor run_scaled_gemm(const Tensor& A, const Tensor& Bt, const Tensor& I,
                       const Tensor& J, const Tensor& alphas,
                       const Tensor& betas) {
  Tensor a = torch::stable::contiguous(A);
  Tensor b = torch::stable::contiguous(Bt);
  Tensor sa = torch::stable::contiguous(alphas);
  Tensor sb = torch::stable::contiguous(betas);
  const int64_t L = a.size(1);
  const int64_t S = L / K * ScaleK;
  STD_TORCH_CHECK(sa.dim() == 2 && sa.size(0) == a.size(0) &&
                      sa.size(1) == S && sb.dim() == 2 &&
                      sb.size(0) == b.size(0) && sb.size(1) == S,
                  "alphas and betas must have ", ScaleK, " scales per ", K,
                  " elements of A's and Bt's rows");
  const T* a_ptr = a.const_data_ptr<T>();
  const T* b_ptr = b.const_data_ptr<T>();
  const T* sa_ptr = sa.const_data_ptr<T>();
  const T* sb_ptr = sb.const_data_ptr<T>();
  return gemm<T, K>(
      a, b, torch::stable::contiguous(I), torch::stable::contiguous(J),
      [=](int64_t i, int64_t j, int64_t chunk, T acc) {
        const auto& a_values =
            unsafe_array_view<T, K>(a_ptr + i * L + chunk * K);
        const auto& b_values =
            unsafe_array_view<T, K>(b_ptr + j * L + chunk * K);
        const T* alpha = sa_ptr + i * S + chunk * ScaleK;
        const T* beta = sb_ptr + j * S + chunk * ScaleK;
        if constexpr (ScaleK == 1) {
          return Model(a_values, b_values, acc, *alpha, *beta);
        } else {
          return Model(a_values, b_values, acc,
                       unsafe_array_view<T, ScaleK>(alpha),
                       unsafe_array_view<T, ScaleK>(beta));
        }
      });
}

}  // namespace fpy_models::runners
