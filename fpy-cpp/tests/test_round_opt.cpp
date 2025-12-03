#include <optional>

#include <fpy.hpp>
#include <gtest/gtest.h>

using namespace fpy;

using round_test_t = std::tuple<exp_t, mant_t, exp_t, mant_t, fpy::RM>;

TEST(RoundOpt, TestRoundWithPrec) {
    std::vector<round_test_t> inputs = {
        // 8 * 2 ** -3 (representable)
        {-3, 8, -1, 2, RM::RNE}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RNA}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTP}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTN}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RTZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        {-3, 8, -1, 2, RM::RAZ}, // 8 * 2 ** -3 => 1 * 2 ** -1
        // 9 * 2 ** -3 (below halfway)
        {-3, 9, -1, 2, RM::RNE}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RNA}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RTP}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 9, -1, 2, RM::RTN}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 2, RM::RTZ}, // 9 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 9, -1, 3, RM::RAZ}, // 9 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 10 * 2 ** -3 (exactly halfway)
        {-3, 10, -1, 2, RM::RNE}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RNA}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 3, RM::RTP}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 10, -1, 2, RM::RTN}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 2, RM::RTZ}, // 10 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 10, -1, 3, RM::RAZ}, // 10 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 11 * 2 ** -3 (above halfway)
        {-3, 11, -1, 3, RM::RNE}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RNA}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 3, RM::RTP}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        {-3, 11, -1, 2, RM::RTN}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 2, RM::RTZ}, // 11 * 2 ** -3 => 1 * 2 ** -1 (down)
        {-3, 11, -1, 3, RM::RAZ}, // 11 * 2 ** -3 => 1 * 3 ** -1 (up)
        // 12 * 2 ** -3 (representable)
        {-3, 12, -1, 3, RM::RNE}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RNA}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTP}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTN}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RTZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
        {-3, 12, -1, 3, RM::RAZ}, // 12 * 2 ** -3 => 1 * 3 ** -1
    };

    for (const auto& [exp_in, c_in, exp_out, c_out, rm] : inputs) {
        const auto x = static_cast<double>(fpy::RealFloat(false, exp_in, c_in));
        const auto y_expect = static_cast<double>(fpy::RealFloat(false, exp_out, c_out));
        const auto y = round_opt::round(x, 2, std::nullopt, rm);
        EXPECT_EQ(y, y_expect);
    }
}
