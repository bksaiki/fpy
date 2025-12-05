#include <fpy.hpp>
#include <gtest/gtest.h>

using namespace fpy;

TEST(MPContext, TestParams) {
    const MPContext ctx(5, fpy::RM::RNE);
    // getters
    EXPECT_EQ(ctx.prec(), 5);
    EXPECT_EQ(ctx.rm(), fpy::RM::RNE);
    // rounding parameters
    EXPECT_EQ(ctx.round_prec(), 5);
    // rounding
    EXPECT_EQ(ctx.round(33.0), 32.0);
}
