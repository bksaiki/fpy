#include <random>

#include <fpy/real.hpp>
#include <gtest/gtest.h>

TEST(RealFloat, TestPrec) {
    /// [[x0]] = 0
    fpy::RealFloat x0;
    EXPECT_EQ(x0.prec(), 0);

    /// [[x1]] = 1 (precision = 1)
    fpy::RealFloat x1(false, 0, 1);
    EXPECT_EQ(x1.prec(), 1);

    /// [[x2]] = 1 (precision = 3)
    fpy::RealFloat x2(false, -2, 4);
    EXPECT_EQ(x2.prec(), 3);

    /// [x3] = 3 (precision = 2)
    fpy::RealFloat x3(false, 0, 3);
    EXPECT_EQ(x3.prec(), 2);
}

TEST(RealFloat, TestEncodeUniform) {
    static constexpr size_t N = 1000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // randomly generate N floating-point values on [-1, 1]
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        const double x = dist(rng);
        const fpy::RealFloat r(x);
        const double y = static_cast<double>(r);
        EXPECT_EQ(x, y);
    }
}

TEST(RealFloat, TestEncodeRepr) {
    static constexpr size_t N = 1000000;

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // randomly generate N bitpatterns for double-precision floats
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    for (size_t i = 0; i < N; i++) {
        const uint64_t b = dist(rng);
        const double x = std::bit_cast<double>(b);
        if (std::isnan(x) || std::isinf(x)) {
            continue;
        }

        const fpy::RealFloat r(x);
        const double y = static_cast<double>(r);
        EXPECT_EQ(x, y);
    }
}
