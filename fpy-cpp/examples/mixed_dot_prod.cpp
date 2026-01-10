#include <chrono>
#include <random>
#include <vector>

#include <fpy.hpp>

using namespace std::chrono;

static double run_reference(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    std::cout << "Computing standard double-precision dot product...\n";
    auto start_ref = high_resolution_clock::now();

    const size_t n = x_vals.size();
    double dot_product = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot_product += x_vals[i] * y_vals[i];
    }

    auto end_ref = high_resolution_clock::now();
    auto duration_ref = duration_cast<microseconds>(end_ref - start_ref).count();

    std::cout << "Reference dot product result: " << dot_product << "\n";
    std::cout << "Duration: " << duration_ref * 1e-6 << " seconds\n\n";
    return duration_ref;
}


static double run_mixed(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals
) {
    // Compute using mixed precision
    const fpy::IEEE754Context ctx_quant(5, 16, fpy::RM::RTZ); // FP16 for quantization
    const fpy::IEEE754Context ctx_mul(8, 19, fpy::RM::RTZ); // TF32 for multiplication
    const fpy::IEEE754Context ctx_add(11, 32, fpy::RM::RNE); // FP32 for addition

    std::cout << "Computing mixed-precision dot product...\n";
    auto start = high_resolution_clock::now();

    const size_t n = x_vals.size();
    double dot_product = 0.0;
    for (size_t i = 0; i < n; i++) {
        // Quantize inputs to FP16
        const auto x_q = fpy::round(x_vals[i], ctx_quant);
        const auto y_q = fpy::round(y_vals[i], ctx_quant);
        // Multiply in TF32 (can be done exactly)
        const auto prod = fpy::mul<fpy::EngineType::EXACT>(x_q, y_q, ctx_mul);
        // Accumulate in FP32
        dot_product = fpy::add(dot_product, prod, ctx_add);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "Mixed-precision dot product result: " << dot_product << "\n";
    std::cout << "Duration: " << duration * 1e-6 << " seconds\n";
    return duration;
}


int main() {
    // Configuration
    static constexpr size_t N = 100'000'000; // 100 million operations

    // Generate random test data in the range [-1.0, 1.0] for the dot product
    std::cout << "Generating random test data...\n";
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::vector<double> x_vals(N);
    std::vector<double> y_vals(N);

    for (size_t i = 0; i < N; i++) {
        x_vals[i] = dist(rng);
        y_vals[i] = dist(rng);
    }

    // Computing standard double-precision dot product for reference
    const double duration_ref = run_reference(x_vals, y_vals);

    // Compute using mixed precision
    const double duration_mixed = run_mixed(x_vals, y_vals);

    std::cout << "Slowdown: " << static_cast<double>(duration_mixed) / duration_ref << "x\n";

    return 0;
}
