#include <algorithm>
#include <array>
#include <cstddef>
#include <thread>
#include <vector>

namespace gemm {

template <class R, class TA, class TB, std::size_t N, class TC>
void dots(R (*fn)(const std::array<TA, N>&, const std::array<TB, N>&, TC),
          const double* A, const double* Bt, long K, const long* I,
          const long* J, double* out, long P, int threads) {
    auto work = [&](long lo, long hi) {
        std::array<TA, N> x;
        std::array<TB, N> y;
        for (long p = lo; p < hi; ++p) {
            const double* a = A + I[p] * K;
            const double* b = Bt + J[p] * K;
            TC c = 0;
            for (long k = 0; k < K; k += N) {
                std::copy(a + k, a + k + N, x.begin());
                std::copy(b + k, b + k + N, y.begin());
                c = fn(x, y, c);
            }
            out[p] = c;
        }
    };
    std::vector<std::thread> pool;
    const long step = (P + threads - 1) / threads;
    for (long lo = 0; lo < P; lo += step)
        pool.emplace_back(work, lo, std::min(P, lo + step));
    for (auto& t : pool) t.join();
}

}  // namespace gemm
