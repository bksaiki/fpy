#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#include <mpfr.h>
#include <fpy.hpp>

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

double benchmark_fpy_mul(const std::vector<double>& x_vals, 
                         const std::vector<double>& y_vals,
                         int p, fpy::RM rm) {
    const size_t n = x_vals.size();
    volatile double result = 0.0; // volatile to prevent optimization
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        result = fpy::mul(x_vals[i], y_vals[i], p, rm);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

double benchmark_mpfr_mul(const std::vector<double>& x_vals,
                          const std::vector<double>& y_vals,
                          int p, fpy::RM rm) {
    const size_t n = x_vals.size();
    mpfr_t mx, my, mr;
    
    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);
    
    volatile double result = 0.0; // volatile to prevent optimization
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < n; i++) {
        mpfr_set_d(mx, x_vals[i], MPFR_RNDN);
        mpfr_set_d(my, y_vals[i], MPFR_RNDN);
        mpfr_mul(mr, mx, my, cvt_rm(rm));
        result = mpfr_get_d(mr, MPFR_RNDN);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    
    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    
    return static_cast<double>(duration) / n; // average time per operation in ns
}

int main() {
    // Configuration
    static constexpr size_t N = 100'000'000; // 10 million operations
    static constexpr int PRECISION = 8;
    static constexpr fpy::RM ROUNDING_MODE = fpy::RM::RNE;
    
    std::cout << "=================================================\n";
    std::cout << "     FPY vs MPFR Multiplication Benchmark\n";
    std::cout << "=================================================\n";
    std::cout << "Operations:     " << N << "\n";
    std::cout << "Precision:      " << PRECISION << " bits\n";
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
    
    for (size_t i = 0; i < N; i++) {
        x_vals[i] = dist(rng);
        y_vals[i] = dist(rng);
    }
    
    std::cout << "Done.\n\n";
    
    // Benchmark FPY
    std::cout << "Benchmarking FPY mul()...\n";
    double fpy_time = benchmark_fpy_mul(x_vals, y_vals, PRECISION, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Benchmark MPFR
    std::cout << "Benchmarking MPFR mpfr_mul()...\n";
    double mpfr_time = benchmark_mpfr_mul(x_vals, y_vals, PRECISION, ROUNDING_MODE);
    std::cout << "Done.\n\n";
    
    // Results
    std::cout << "=================================================\n";
    std::cout << "                   RESULTS\n";
    std::cout << "=================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "FPY mul():         " << fpy_time << " ns/op\n";
    std::cout << "MPFR mpfr_mul():   " << mpfr_time << " ns/op\n";
    std::cout << "-------------------------------------------------\n";
    
    if (fpy_time < mpfr_time) {
        double speedup = mpfr_time / fpy_time;
        std::cout << "FPY is " << std::setprecision(2) << speedup << "x FASTER than MPFR\n";
    } else {
        double slowdown = fpy_time / mpfr_time;
        std::cout << "FPY is " << std::setprecision(2) << slowdown << "x SLOWER than MPFR\n";
    }
    std::cout << "=================================================\n";
    
    return 0;
}
