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

namespace fpy_models::model_nv_volta_f16_f32 {

float dpa_special_values__5902d066__f17da570__bccc3234d(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    std::array<bool, 4> t10 = std::array<bool, 4>{};
    for (int8_t t11 = 0; t11 < 4; ++t11) {
        float a = A[static_cast<size_t>(t11)];
        t10[static_cast<size_t>(t11)] = std::isnan(a);
    }
    std::array<bool, 4> t13 = std::array<bool, 4>{};
    for (int8_t t14 = 0; t14 < 4; ++t14) {
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
    for (int8_t _i = 0; _i < 4; ++_i) {
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

float exponent__5902d066__67169c42__bf0928713(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

float exponent__5902d066__5ebbc1f4__ba49bfc41(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

double fused_sum__5902d066__d4392ee9__v99d6777c__b25785cc5(const std::array<float, 5>& xs, float n) {
    std::array<double, 5> ts = std::array<double, 5>{};
    for (int8_t t6 = 0; t6 < 5; ++t6) {
        float x = xs[static_cast<size_t>(t6)];
        float _k = (n + static_cast<float>(1));
        auto&& _tmp1 = (-_k);
        auto&& _tmp2 = static_cast<double>(x);
        double _t = (std::isfinite(_tmp1) ? std::ldexp(_tmp2, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * _tmp2);
        assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
        float _tmp3 = std::trunc(_t);
        assert((std::fabs(_tmp3) <= 33554432) && "fpy: overflow occurred so rounding is undefined");
        float _t9 = _tmp3;
        auto&& _tmp4 = static_cast<double>(_t9);
        double t8 = (std::isfinite(_k) ? std::ldexp(_tmp4, static_cast<int>(_k)) : std::pow(2.0, _k) * _tmp4);
        ts[static_cast<size_t>(t6)] = t8;
    }
    return std::accumulate(ts.begin() + 1, ts.end(), ts[static_cast<size_t>(0)]);
}

std::array<float, 5> join__5902d066__5c7f8d3f__b31933400(const std::array<float, 4>& xs, const std::array<float, 1>& ys) {
    std::array<float, 5> zs = std::array<float, 5>{};
    for (int8_t i = 0; i < 4; ++i) {
        zs[static_cast<size_t>(i)] = xs[static_cast<size_t>(i)];
    }
    for (int8_t i_1 = 0; i_1 < 1; ++i_1) {
        zs[static_cast<size_t>((4 + i_1))] = ys[static_cast<size_t>(i_1)];
    }
    return zs;
}

float fdpa_round__5902d066__ce25623c__v5af88987__bfe20c608(double s) {
    if ((std::fabs(s) >= 3.402823669209385e+38)) {
        float t{};
        if (std::signbit(s)) {
            t = (-std::numeric_limits<float>::infinity());
        } else {
            t = std::numeric_limits<float>::infinity();
        }
        return t;
    }
    const auto _tmp1 = std::fegetround();
    std::fesetround(FE_TOWARDZERO);
    float _tmp2 = static_cast<float>(s);
    std::fesetround(_tmp1);
    return _tmp2;
    std::fesetround(_tmp1);
}

float t_fdpa__5902d066__f17da570__b7c8363e5(const std::array<float, 4>& A, const std::array<float, 4>& B, float c) {
    std::array<uint8_t, 4> _tmp1 = std::array<uint8_t, 4>{};
    std::iota(_tmp1.begin(), _tmp1.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 4> t32 = _tmp1;
    std::array<std::tuple<float, float>, 4> t = std::array<std::tuple<float, float>, 4>{};
    for (int8_t t33 = 0; t33 < 4; ++t33) {
        uint8_t t30 = t32[static_cast<size_t>(t33)];
        t[static_cast<size_t>(t33)] = std::make_tuple(A[static_cast<size_t>(t30)], B[static_cast<size_t>(t30)]);
    }
    std::array<float, 4> prods = std::array<float, 4>{};
    for (int8_t t23 = 0; t23 < 4; ++t23) {
        auto&& _tmp2 = t[static_cast<size_t>(t23)];
        float a = std::get<0>(_tmp2);
        float b = std::get<1>(_tmp2);
        prods[static_cast<size_t>(t23)] = (a * b);
    }
    std::array<bool, 4> t25 = std::array<bool, 4>{};
    for (int8_t t26 = 0; t26 < 4; ++t26) {
        float p = prods[static_cast<size_t>(t26)];
        t25[static_cast<size_t>(t26)] = (!std::isfinite(p));
    }
    bool t24 = std::any_of(t25.begin(), t25.end(), [](bool _tmp3) { return _tmp3; });
    if ((!t24)) {
        t24 = (!std::isfinite(c));
    }
    if (t24) {
        return dpa_special_values__5902d066__f17da570__bccc3234d(A, B, c);
    }
    std::array<uint8_t, 4> _tmp4 = std::array<uint8_t, 4>{};
    std::iota(_tmp4.begin(), _tmp4.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 4> t34 = _tmp4;
    std::array<std::tuple<float, float, float>, 4> t27 = std::array<std::tuple<float, float, float>, 4>{};
    for (int8_t t35 = 0; t35 < 4; ++t35) {
        uint8_t t31 = t34[static_cast<size_t>(t35)];
        t27[static_cast<size_t>(t35)] = std::make_tuple(prods[static_cast<size_t>(t31)], A[static_cast<size_t>(t31)], B[static_cast<size_t>(t31)]);
    }
    std::array<float, 4> es = std::array<float, 4>{};
    for (int8_t t28 = 0; t28 < 4; ++t28) {
        auto&& _tmp5 = t27[static_cast<size_t>(t28)];
        float p_1 = std::get<0>(_tmp5);
        float a_1 = std::get<1>(_tmp5);
        float b_1 = std::get<2>(_tmp5);
        float t29{};
        if ((p_1 == static_cast<float>(0))) {
            t29 = -131;
        } else {
            t29 = (exponent__5902d066__67169c42__bf0928713(a_1, -14) + exponent__5902d066__67169c42__bf0928713(b_1, -14));
        }
        es[static_cast<size_t>(t28)] = t29;
    }
    float e_c{};
    if ((c == static_cast<float>(0))) {
        e_c = -131;
    } else {
        e_c = exponent__5902d066__5ebbc1f4__ba49bfc41(c, -126);
    }
    float _tmp6 = es[0];
    for (size_t _tmp7 = 1; _tmp7 < es.size(); ++_tmp7) {
        auto&& _tmp8 = es[_tmp7];
        _tmp6 = ((std::isnan(_tmp6) || std::isnan(_tmp8)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp6 < _tmp8 || (_tmp6 == _tmp8 && std::signbit(_tmp6))) ? _tmp8 : _tmp6));
    }
    float e_max = ((std::isnan(_tmp6) || std::isnan(e_c)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp6 < e_c || (_tmp6 == e_c && std::signbit(_tmp6))) ? e_c : _tmp6));
    double s = fused_sum__5902d066__d4392ee9__v99d6777c__b25785cc5(join__5902d066__5c7f8d3f__b31933400(prods, std::array<float, 1>{{c}}), ((e_max - static_cast<float>(23)) - static_cast<float>(1)));
    return fdpa_round__5902d066__ce25623c__v5af88987__bfe20c608(s);
}

float t_fdpa_chain(const std::array<float, 8>& A, const std::array<float, 8>& B, float c) {
    float d = c;
    for (int8_t i = 0; i < 8; i += 4) {
        std::array<float, 4> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 4)), _tmp1.begin());
        std::array<float, 4> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 4)), _tmp2.begin());
        d = t_fdpa__5902d066__f17da570__b7c8363e5(_tmp1, _tmp2, d);
    }
    return d;
}
}
