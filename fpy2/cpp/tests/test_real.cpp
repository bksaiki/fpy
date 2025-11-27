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
