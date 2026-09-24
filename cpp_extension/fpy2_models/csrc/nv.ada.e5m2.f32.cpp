#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>
#include <tuple>

namespace fpy_models::model_nv_ada_e5m2_f32 {

float dpa_special_values__5902d066__56c576e1__be898a2fb(const std::array<float, 16>& A, const std::array<float, 16>& B, float c) {
    std::array<bool, 16> t10 = std::array<bool, 16>{};
    for (int8_t t11 = 0; t11 < 16; ++t11) {
        float a = A[static_cast<size_t>(t11)];
        t10[static_cast<size_t>(t11)] = std::isnan(a);
    }
    std::array<bool, 16> t13 = std::array<bool, 16>{};
    for (int8_t t14 = 0; t14 < 16; ++t14) {
        float b = B[static_cast<size_t>(t14)];
        t13[static_cast<size_t>(t14)] = std::isnan(b);
    }
    bool t16 = std::any_of(t10.begin(), t10.end(), [](bool _tmp1) { return _tmp1; });
    if ((!t16)) {
        t16 = std::any_of(t13.begin(), t13.end(), [](bool _tmp2) { return _tmp2; });
    }
    if ((!t16)) {
        t16 = std::isnan(c);
    }
    if (t16) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    bool has_inf = std::isinf(c);
    bool inf_sgn = std::signbit(c);
    for (int8_t _i = 0; _i < 16; ++_i) {
        float a_1 = A[static_cast<size_t>(_i)];
        float b_1 = B[static_cast<size_t>(_i)];
        bool t17 = std::isinf(a_1);
        if ((!t17)) {
            t17 = std::isinf(b_1);
        }
        if (t17) {
            float t = (a_1 * b_1);
            if (std::isnan(t)) {
                return t;
            }
            bool t_sgn = std::signbit(t);
            if (has_inf) {
                if ((inf_sgn != t_sgn)) {
                    return std::numeric_limits<float>::quiet_NaN();
                }
            } else {
                has_inf = true;
                inf_sgn = t_sgn;
            }
        }
    }
    assert(has_inf && "fpy assert: 'expected either NaN or infinity in the input'");
    float t18{};
    if (inf_sgn) {
        t18 = (-std::numeric_limits<float>::infinity());
    } else {
        t18 = std::numeric_limits<float>::infinity();
    }
    return t18;
}

float exponent__5902d066__68cd2a5b__b22b41228(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

float exponent__5902d066__5ebbc1f4__ba49bfc41(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

double fused_sum__5902d066__f7d46888__v99d6777c__b17e25a32(const std::array<float, 17>& xs, float n) {
    std::array<double, 17> ts = std::array<double, 17>{};
    for (int8_t t6 = 0; t6 < 17; ++t6) {
        float x = xs[static_cast<size_t>(t6)];
        float _k = (n + static_cast<float>(1));
        auto&& _tmp1 = (-_k);
        auto&& _tmp2 = static_cast<double>(x);
        double _t = (std::isfinite(_tmp1) ? std::ldexp(_tmp2, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * _tmp2);
        assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
        float _tmp3 = std::trunc(_t);
        assert((std::fabs(_tmp3) <= 32768) && "fpy: overflow occurred so rounding is undefined");
        float _t9 = _tmp3;
        auto&& _tmp4 = static_cast<double>(_t9);
        double t8 = (std::isfinite(_k) ? std::ldexp(_tmp4, static_cast<int>(_k)) : std::pow(2.0, _k) * _tmp4);
        ts[static_cast<size_t>(t6)] = t8;
    }
    return std::accumulate(ts.begin() + 1, ts.end(), ts[static_cast<size_t>(0)]);
}

std::array<float, 17> join__5902d066__1ffa361a__ba005ae93(const std::array<float, 16>& xs, const std::array<float, 1>& ys) {
    std::array<float, 17> zs = std::array<float, 17>{};
    for (int8_t i = 0; i < 16; ++i) {
        zs[static_cast<size_t>(i)] = xs[static_cast<size_t>(i)];
    }
    for (int8_t i_1 = 0; i_1 < 1; ++i_1) {
        zs[static_cast<size_t>((16 + i_1))] = ys[static_cast<size_t>(i_1)];
    }
    return zs;
}

double fdpa_round__5902d066__272ffd9e__vae2afb49__b75e9aea6(double s) {
    if ((std::fabs(s) >= 3.402823669209385e+38)) {
        float t{};
        if (std::signbit(s)) {
            t = (-std::numeric_limits<float>::infinity());
        } else {
            t = std::numeric_limits<float>::infinity();
        }
        return t;
    }
    double t2{};
    if (std::isnan(s)) {
        t2 = std::numeric_limits<float>::quiet_NaN();
    } else if (std::isinf(s)) {
        if (std::signbit(s)) {
            t2 = (-std::numeric_limits<float>::infinity());
        } else {
            t2 = std::numeric_limits<float>::infinity();
        }
    } else if ((s == static_cast<double>(0))) {
        t2 = (std::signbit(s) ? -0.0 : static_cast<float>(0));
    } else if ((s >= 3.402823669209385e+38)) {
        t2 = 3.4026159773350432e+38;
    } else if ((s <= -3.402823669209385e+38)) {
        t2 = -3.4026159773350432e+38;
    } else {
        double t3{};
        if ((std::fabs(s) < static_cast<double>(1.1754943508222875e-38))) {
            float _t = static_cast<float>((6.96898287454082e+41 * s));
            float _tmp1 = std::trunc(_t);
            assert((std::fabs(_tmp1) <= 8192) && "fpy: overflow occurred so rounding is undefined");
            float _t6 = _tmp1;
            t3 = (1.4349296274686127e-42 * _t6);
        } else {
            int16_t exp = (static_cast<int16_t>(std::ilogb(s)) - static_cast<int16_t>(13));
            auto&& _tmp2 = (-exp);
            double _t7 = std::ldexp(s, static_cast<int>(_tmp2));
            float _tmp3 = std::trunc(_t7);
            assert((std::fabs(_tmp3) <= 16384) && "fpy: overflow occurred so rounding is undefined");
            float _t8 = _tmp3;
            t3 = std::ldexp(static_cast<double>(_t8), static_cast<int>(exp));
        }
        if ((t3 > static_cast<double>(3.4026159773350432e+38))) {
            t2 = 3.4026159773350432e+38;
        } else if ((t3 < static_cast<double>(-3.4026159773350432e+38))) {
            t2 = -3.4026159773350432e+38;
        } else {
            t2 = t3;
        }
    }
    return t2;
}

double t_fdpa__5902d066__56c576e1__be41fc96e(const std::array<float, 16>& A, const std::array<float, 16>& B, float c) {
    std::array<uint8_t, 16> _tmp1 = std::array<uint8_t, 16>{};
    std::iota(_tmp1.begin(), _tmp1.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 16> t32 = _tmp1;
    std::array<std::tuple<float, float>, 16> t = std::array<std::tuple<float, float>, 16>{};
    for (int8_t t33 = 0; t33 < 16; ++t33) {
        uint8_t t30 = t32[static_cast<size_t>(t33)];
        t[static_cast<size_t>(t33)] = std::make_tuple(A[static_cast<size_t>(t30)], B[static_cast<size_t>(t30)]);
    }
    std::array<float, 16> prods = std::array<float, 16>{};
    for (int8_t t23 = 0; t23 < 16; ++t23) {
        auto&& _tmp2 = t[static_cast<size_t>(t23)];
        float a = std::get<0>(_tmp2);
        float b = std::get<1>(_tmp2);
        prods[static_cast<size_t>(t23)] = (a * b);
    }
    std::array<bool, 16> t25 = std::array<bool, 16>{};
    for (int8_t t26 = 0; t26 < 16; ++t26) {
        float p = prods[static_cast<size_t>(t26)];
        t25[static_cast<size_t>(t26)] = (!std::isfinite(p));
    }
    bool t24 = std::any_of(t25.begin(), t25.end(), [](bool _tmp3) { return _tmp3; });
    if ((!t24)) {
        t24 = (!std::isfinite(c));
    }
    if (t24) {
        return dpa_special_values__5902d066__56c576e1__be898a2fb(A, B, c);
    }
    std::array<uint8_t, 16> _tmp4 = std::array<uint8_t, 16>{};
    std::iota(_tmp4.begin(), _tmp4.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 16> t34 = _tmp4;
    std::array<std::tuple<float, float, float>, 16> t27 = std::array<std::tuple<float, float, float>, 16>{};
    for (int8_t t35 = 0; t35 < 16; ++t35) {
        uint8_t t31 = t34[static_cast<size_t>(t35)];
        t27[static_cast<size_t>(t35)] = std::make_tuple(prods[static_cast<size_t>(t31)], A[static_cast<size_t>(t31)], B[static_cast<size_t>(t31)]);
    }
    std::array<float, 16> es = std::array<float, 16>{};
    for (int8_t t28 = 0; t28 < 16; ++t28) {
        auto&& _tmp5 = t27[static_cast<size_t>(t28)];
        float p_1 = std::get<0>(_tmp5);
        float a_1 = std::get<1>(_tmp5);
        float b_1 = std::get<2>(_tmp5);
        float t29{};
        if ((p_1 == static_cast<float>(0))) {
            t29 = -132;
        } else {
            t29 = (exponent__5902d066__68cd2a5b__b22b41228(a_1, -14) + exponent__5902d066__68cd2a5b__b22b41228(b_1, -14));
        }
        es[static_cast<size_t>(t28)] = t29;
    }
    float e_c{};
    if ((c == static_cast<float>(0))) {
        e_c = -132;
    } else {
        e_c = exponent__5902d066__5ebbc1f4__ba49bfc41(c, -126);
    }
    float _tmp6 = es[0];
    for (size_t _tmp7 = 1; _tmp7 < es.size(); ++_tmp7) {
        auto&& _tmp8 = es[_tmp7];
        _tmp6 = ((std::isnan(_tmp6) || std::isnan(_tmp8)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp6 < _tmp8 || (_tmp6 == _tmp8 && std::signbit(_tmp6))) ? _tmp8 : _tmp6));
    }
    float e_max = ((std::isnan(_tmp6) || std::isnan(e_c)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp6 < e_c || (_tmp6 == e_c && std::signbit(_tmp6))) ? e_c : _tmp6));
    double s = fused_sum__5902d066__f7d46888__v99d6777c__b17e25a32(join__5902d066__1ffa361a__ba005ae93(prods, std::array<float, 1>{{c}}), ((e_max - static_cast<float>(13)) - static_cast<float>(1)));
    return fdpa_round__5902d066__272ffd9e__vae2afb49__b75e9aea6(s);
}

double t_fdpa_chain(const std::array<float, 16>& A, const std::array<float, 16>& B, float c) {
    double d = c;
    for (int8_t i = 0; i < 16; i += 16) {
        std::array<float, 16> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 16)), _tmp1.begin());
        std::array<float, 16> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 16)), _tmp2.begin());
        d = t_fdpa__5902d066__56c576e1__be41fc96e(_tmp1, _tmp2, d);
    }
    return d;
}
}
