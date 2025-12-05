#include <fpy.hpp>
#include <gtest/gtest.h>

using namespace fpy;

TEST(Context, TestMPContext) {
    const MPContext ctx(5, fpy::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 5);
    EXPECT_EQ(ctx.rm(), fpy::RM::RNE);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 5);
    // rounding
    EXPECT_EQ(ctx.round(33.0), 32.0);
}

TEST(Context, TestMPSContext) {
    const MPSContext ctx(5, -5, fpy::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 5);
    EXPECT_EQ(ctx.emin(), -5);
    EXPECT_EQ(ctx.rm(), fpy::RM::RNE);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 5);
    EXPECT_EQ(ctx.n(), -10);
    // rounding
    EXPECT_EQ(ctx.round(33.0), 32.0);
    EXPECT_EQ(ctx.round(0.00048828125), 0.0);
}
