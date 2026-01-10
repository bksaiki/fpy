#include <mpfr.h>
#include <random>

#include <fpy.hpp>
#include <gtest/gtest.h>

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

static double ref_add(double x, double y, int p, fpy::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_add(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_sub(double x, double y, int p, fpy::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_sub(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_mul(double x, double y, int p, fpy::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_mul(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_div(double x, double y, int p, fpy::RM rm) {
    mpfr_t mx, my, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_div(mr, mx, my, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mr);
    return r;
}

static double ref_sqrt(double x, int p, fpy::RM rm) {
    mpfr_t mx, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_sqrt(mr, mx, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(mr);
    return r;
}

static double ref_fma(double x, double y, double z, int p, fpy::RM rm) {
    mpfr_t mx, my, mz, mr;
    double r;

    mpfr_init2(mx, 53);
    mpfr_init2(my, 53);
    mpfr_init2(mz, 53);
    mpfr_init2(mr, p);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_set_d(my, y, MPFR_RNDN);
    mpfr_set_d(mz, z, MPFR_RNDN);
    mpfr_fma(mr, mx, my, mz, cvt_rm(rm));

    r = mpfr_get_d(mr, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(my);
    mpfr_clear(mz);
    mpfr_clear(mr);
    return r;
}


TEST(OpsFloat, TestAddUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<fpy::RM> rounding_modes = {
        fpy::RM::RNE,
        fpy::RM::RTP,
        fpy::RM::RTN,
        fpy::RM::RTZ,
        fpy::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const fpy::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_add(x, y, p, rm);
                const double z = fpy::add(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestSubUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<fpy::RM> rounding_modes = {
        fpy::RM::RNE,
        fpy::RM::RTP,
        fpy::RM::RTN,
        fpy::RM::RTZ,
        fpy::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const fpy::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_sub(x, y, p, rm);
                const double z = fpy::sub(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestMulUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<fpy::RM> rounding_modes = {
        fpy::RM::RNE,
        fpy::RM::RTP,
        fpy::RM::RTN,
        fpy::RM::RTZ,
        fpy::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const fpy::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                const double z_ref = ref_mul(x, y, p, rm);
                const double z = fpy::mul(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestDivUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<fpy::RM> rounding_modes = {
        fpy::RM::RNE,
        fpy::RM::RTP,
        fpy::RM::RTN,
        fpy::RM::RTZ,
        fpy::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const fpy::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);

                // skip division by values close to zero
                if (std::abs(y) < 1e-10) continue;

                const double z_ref = ref_div(x, y, p, rm);
                const double z = fpy::div(x, y, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestSqrtUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<fpy::RM> rounding_modes = {
        fpy::RM::RNE,
        fpy::RM::RTP,
        fpy::RM::RTN,
        fpy::RM::RTZ,
        fpy::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const fpy::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [0, 1]
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);

                const double z_ref = ref_sqrt(x, p, rm);
                const double z = fpy::sqrt(x, ctx);
                EXPECT_EQ(z_ref, z);
            }
        }
    }
}

TEST(OpsFloat, TestFmaUniform) {
    static constexpr size_t N = 1000000;

    // rounding modes to test
    const std::vector<fpy::RM> rounding_modes = {
        fpy::RM::RNE,
        fpy::RM::RTP,
        fpy::RM::RTN,
        fpy::RM::RTZ,
        fpy::RM::RAZ,
    };

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    // sweep over precisions from 2 to 8
    for (int p = 2; p <= 8; p++) {
        // sweep over rounding modes
        for (const auto rm : rounding_modes) {
            // rounding context
            const fpy::MPContext ctx(p, rm);

            // randomly generate N floating-point values on [-1, 1]
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < N; i++) {
                const double x = dist(rng);
                const double y = dist(rng);
                const double z = dist(rng);

                const double w_ref = ref_fma(x, y, z, p, rm);
                const double w = fpy::fma(x, y, z, ctx);
                EXPECT_EQ(w_ref, w);
            }
        }
    }
}
