#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#include <mpfr.h>
#include <fpy.hpp>

// SoftFloat integration
extern "C" {
    #include "softfloat.h"
}

using namespace std::chrono;

static mpfr_rnd_t cvt_rm(fpy::RM rm) {
    switch (rm) {
        case fpy::RM::RNE:
            return MPFR_RNDN;
        case fpy::RM::RTP:
            return MPFR_RNDU;
        case fpy::RM::RTN:
            return MPFR_RNDD;
        case fpy::RM::RTZ:
            return MPFR_RNDZ;
        case fpy::RM::RAZ:
            return MPFR_RNDA;
        default:
            throw std::runtime_error("invalid rounding mode");
    }
}

static std::string rm_to_string(fpy::RM rm) {
    switch (rm) {
        case fpy::RM::RNE:
            return "RNE (Round to Nearest Even)";
        case fpy::RM::RTP:
            return "RTP (Round Toward Positive)";
        case fpy::RM::RTN:
            return "RTN (Round Toward Negative)";
        case fpy::RM::RTZ:
            return "RTZ (Round to Zero)";
        case fpy::RM::RAZ:
            return "RAZ (Round Away from Zero)";
        default:
            return "Unknown";
    }
}

// SoftFloat rounding mode conversion
static void set_softfloat_rm(fpy::RM rm) {
    switch (rm) {
        case fpy::RM::RNE:
            softfloat_roundingMode = softfloat_round_near_even;
            break;
        case fpy::RM::RTP:
            softfloat_roundingMode = softfloat_round_max;
            break;
        case fpy::RM::RTN:
            softfloat_roundingMode = softfloat_round_min;
            break;
        case fpy::RM::RTZ:
            softfloat_roundingMode = softfloat_round_minMag;
            break;
        case fpy::RM::RAZ:
            softfloat_roundingMode = softfloat_round_near_maxMag;
            break;
        default:
            throw std::runtime_error("invalid rounding mode");
    }
}

double benchmark_fpy_fma(const std::vector<double>& x_vals, 
                         const std::vector<double>& y_vals,
                         const std::vector<double>& z_vals,
                         int p, fpy::RM rm) {
    const size_t n = x_vals.size();
    volatile double result = 0.0; // volatile to prevent optimization

    const fpy::MPContext ctx(p, rm);
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        result = fpy::fma(x_vals[i], y_vals[i], z_vals[i], ctx);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

double benchmark_mpfr_fma(const std::vector<double>& x_vals,
                          const std::vector<double>& y_vals,
                          const std::vector<double>& z_vals,
                          int p, fpy::RM rm) {
    const size_t n = x_vals.size();
    mpfr_t mx, my, mz, mr;
    
    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mz, 53);
    mpfr_init2(mr, p);
    
    volatile double result = 0.0; // volatile to prevent optimization
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_d(my, y_vals[i], MPFR_RNDN);
        mpfr_set_d(mz, z_vals[i], MPFR_RNDN);
        mpfr_fma(mr, mx, my, mz, cvt_rm(rm));
        result = mpfr_get_d(mr, MPFR_RNDN);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    
    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mz);
    mpfr_clear(mr);
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

// SoftFloat benchmark (32-bit)
double benchmark_softfloat_fma(const std::vector<double>& x_vals,
                              const std::vector<double>& y_vals,
                              const std::vector<double>& z_vals,
                              fpy::RM rm) {
    const size_t n = x_vals.size();
    
    set_softfloat_rm(rm);
    volatile double result = 0.0; // volatile to prevent optimization
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        // Convert doubles to SoftFloat f32, compute fma, convert back
        union { float f; uint32_t i; } converter_x, converter_y, converter_z, converter_result;
        converter_x.f = static_cast<float>(x_vals[i]);
        converter_y.f = static_cast<float>(y_vals[i]);
        converter_z.f = static_cast<float>(z_vals[i]);
        float32_t sf_x = { converter_x.i };
        float32_t sf_y = { converter_y.i };
        float32_t sf_z = { converter_z.i };
        
        float32_t sf_result = f32_mulAdd(sf_x, sf_y, sf_z);
        
        converter_result.i = sf_result.v;
        result = static_cast<double>(converter_result.f);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    (void) result; // prevent unused variable warning
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

int main() {
    // Configuration
    static constexpr size_t N = 100'000'000; // 10 million operations
    static constexpr int PRECISION = 24;
    static constexpr fpy::RM ROUNDING_MODE = fpy::RM::RNE;
    
    std::cout << "=======================================================\n";
    std::cout << "   FPY vs MPFR vs SoftFloat Fused Multiply-Add Benchmark\n";
    std::cout << "=======================================================\n";
    std::cout << "Operations:     " << N << "\n";
    std::cout << "Precision:      " << PRECISION << " bits (FPY/MPFR), 32-bit (SoftFloat)\n";
    std::cout << "Rounding mode:  " << rm_to_string(ROUNDING_MODE) << "\n";
    std::cout << "Input range:    [-1.0, 1.0] (uniform)\n";
    std::cout << "-------------------------------------------------\n\n";
    
    // Generate random test data
    std::cout << "Generating random test data...\n";
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::vector<double> x_vals(N);
    std::vector<double> y_vals(N);
    std::vector<double> z_vals(N);
    
    for (size_t i = 0; i < N; i++) {
        x_vals[i] = dist(rng);
        y_vals[i] = dist(rng);
        z_vals[i] = dist(rng);
    }
    
    std::cout << "Done.\n\n";
    
    // Benchmark FPY
    std::cout << "Benchmarking FPY fma()...\n";
    double fpy_time = benchmark_fpy_fma(x_vals, y_vals, z_vals, PRECISION, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Benchmark MPFR
    std::cout << "Benchmarking MPFR mpfr_fma()...\n";
    double mpfr_time = benchmark_mpfr_fma(x_vals, y_vals, z_vals, PRECISION, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Benchmark SoftFloat
    std::cout << "Benchmarking SoftFloat f32_mulAdd()...\n";
    double softfloat_time = benchmark_softfloat_fma(x_vals, y_vals, z_vals, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Results
    std::cout << "=======================================================\n";
    std::cout << "                      RESULTS\n";
    std::cout << "=======================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "FPY fma():               " << fpy_time << " ns/op\n";
    std::cout << "MPFR mpfr_fma():         " << mpfr_time << " ns/op\n";
    std::cout << "SoftFloat f32_mulAdd():  " << softfloat_time << " ns/op\n";
    std::cout << "-------------------------------------------------------\n";
    
    // Performance comparisons
    if (fpy_time < mpfr_time) {
        double speedup = mpfr_time / fpy_time;
        std::cout << "FPY is " << std::setprecision(2) << speedup << "x FASTER than MPFR\n";
    } else {
        double slowdown = fpy_time / mpfr_time;
        std::cout << "FPY is " << std::setprecision(2) << slowdown << "x SLOWER than MPFR\n";
    }
    
    if (fpy_time < softfloat_time) {
        double speedup = softfloat_time / fpy_time;
        std::cout << "FPY is " << std::setprecision(2) << speedup << "x FASTER than SoftFloat\n";
    } else {
        double slowdown = fpy_time / softfloat_time;
        std::cout << "FPY is " << std::setprecision(2) << slowdown << "x SLOWER than SoftFloat\n";
    }
    
    if (mpfr_time < softfloat_time) {
        double speedup = softfloat_time / mpfr_time;
        std::cout << "MPFR is " << std::setprecision(2) << speedup << "x FASTER than SoftFloat\n";
    } else {
        double slowdown = mpfr_time / softfloat_time;
        std::cout << "MPFR is " << std::setprecision(2) << slowdown << "x SLOWER than SoftFloat\n";
    }
    std::cout << "=======================================================\n";
    
    return 0;
}
