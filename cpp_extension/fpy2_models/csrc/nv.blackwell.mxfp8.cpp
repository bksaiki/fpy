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

namespace fpy_models::model_nv_blackwell_mxfp8 {

float dpa_special_values__5902d066__5bb08766__b8c6e33c0(const std::array<float, 32>& A, const std::array<float, 32>& B, float c) {
    std::array<bool, 32> t10 = std::array<bool, 32>{};
    for (int8_t t11 = 0; t11 < 32; ++t11) {
        float a = A[static_cast<size_t>(t11)];
        t10[static_cast<size_t>(t11)] = std::isnan(a);
    }
    std::array<bool, 32> t13 = std::array<bool, 32>{};
    for (int8_t t14 = 0; t14 < 32; ++t14) {
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
    for (int8_t _i = 0; _i < 32; ++_i) {
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

float exponent__5902d066__1c39a6a7__b0c313474(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

float exponent__5902d066__5ebbc1f4__ba49bfc41(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

double fused_sum__5902d066__dcba4d50__v99d6777c__b81540742(const std::array<double, 33>& xs, float n) {
    std::array<double, 33> ts = std::array<double, 33>{};
    for (int8_t t6 = 0; t6 < 33; ++t6) {
        double x = xs[static_cast<size_t>(t6)];
        float _k = (n + static_cast<float>(1));
        auto&& _tmp1 = (-_k);
        double _t = (std::isfinite(_tmp1) ? std::ldexp(x, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * x);
        assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
        float _tmp2 = std::trunc(_t);
        assert((std::fabs(_tmp2) <= 536870912) && "fpy: overflow occurred so rounding is undefined");
        float _t9 = _tmp2;
        auto&& _tmp3 = static_cast<double>(_t9);
        double t8 = (std::isfinite(_k) ? std::ldexp(_tmp3, static_cast<int>(_k)) : std::pow(2.0, _k) * _tmp3);
        ts[static_cast<size_t>(t6)] = t8;
    }
    return std::accumulate(ts.begin() + 1, ts.end(), ts[static_cast<size_t>(0)]);
}

std::array<double, 33> join__5902d066__d0e61c18__b5e34ac9d(const std::array<double, 32>& xs, const std::array<float, 1>& ys) {
    std::array<double, 33> zs = std::array<double, 33>{};
    for (int8_t i = 0; i < 32; ++i) {
        zs[static_cast<size_t>(i)] = xs[static_cast<size_t>(i)];
    }
    for (int8_t i_1 = 0; i_1 < 1; ++i_1) {
        zs[static_cast<size_t>((32 + i_1))] = ys[static_cast<size_t>(i_1)];
    }
    return zs;
}

float fdpa_round__5902d066__c58205ca__v5af88987__bc5f10b0e(double s) {
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

float st_fdpa(const std::array<float, 32>& A, const std::array<float, 32>& B, float c, float alpha, float beta) {
    bool t27 = std::isnan(alpha);
    if ((!t27)) {
        t27 = std::isnan(beta);
    }
    if (t27) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::array<int8_t, 32> _tmp1 = std::array<int8_t, 32>{};
    std::iota(_tmp1.begin(), _tmp1.end(), static_cast<int8_t>(0));
    std::array<int8_t, 32> t36 = _tmp1;
    std::array<std::tuple<float, float>, 32> t = std::array<std::tuple<float, float>, 32>{};
    for (int8_t t37 = 0; t37 < 32; ++t37) {
        int8_t t34 = t36[static_cast<size_t>(t37)];
        t[static_cast<size_t>(t37)] = std::make_tuple(A[static_cast<size_t>(t34)], B[static_cast<size_t>(t34)]);
    }
    std::array<double, 32> prods = std::array<double, 32>{};
    for (int8_t t26 = 0; t26 < 32; ++t26) {
        auto&& _tmp2 = t[static_cast<size_t>(t26)];
        float a = std::get<0>(_tmp2);
        float b = std::get<1>(_tmp2);
        prods[static_cast<size_t>(t26)] = ((static_cast<double>((a * b)) * static_cast<double>(alpha)) * static_cast<double>(beta));
    }
    std::array<bool, 32> t28 = std::array<bool, 32>{};
    for (int8_t t29 = 0; t29 < 32; ++t29) {
        double p = prods[static_cast<size_t>(t29)];
        t28[static_cast<size_t>(t29)] = (!std::isfinite(p));
    }
    bool t32 = std::any_of(t28.begin(), t28.end(), [](bool _tmp3) { return _tmp3; });
    if ((!t32)) {
        t32 = (!std::isfinite(c));
    }
    if (t32) {
        return dpa_special_values__5902d066__5bb08766__b8c6e33c0(A, B, c);
    }
    std::array<int8_t, 32> _tmp4 = std::array<int8_t, 32>{};
    std::iota(_tmp4.begin(), _tmp4.end(), static_cast<int8_t>(0));
    std::array<int8_t, 32> t38 = _tmp4;
    std::array<std::tuple<double, float, float>, 32> t30 = std::array<std::tuple<double, float, float>, 32>{};
    for (int8_t t39 = 0; t39 < 32; ++t39) {
        int8_t t35 = t38[static_cast<size_t>(t39)];
        t30[static_cast<size_t>(t39)] = std::make_tuple(prods[static_cast<size_t>(t35)], A[static_cast<size_t>(t35)], B[static_cast<size_t>(t35)]);
    }
    std::array<float, 32> es = std::array<float, 32>{};
    for (int8_t t31 = 0; t31 < 32; ++t31) {
        auto&& _tmp5 = t30[static_cast<size_t>(t31)];
        double p_1 = std::get<0>(_tmp5);
        float a_1 = std::get<1>(_tmp5);
        float b_1 = std::get<2>(_tmp5);
        float t33{};
        if ((p_1 == static_cast<double>(0))) {
            t33 = -133;
        } else {
            t33 = (((exponent__5902d066__68cd2a5b__b22b41228(a_1, -14) + exponent__5902d066__68cd2a5b__b22b41228(b_1, -14)) + exponent__5902d066__1c39a6a7__b0c313474(alpha, -127)) + exponent__5902d066__1c39a6a7__b0c313474(beta, -127));
        }
        es[static_cast<size_t>(t31)] = t33;
    }
    float e_c{};
    if ((c == static_cast<float>(0))) {
        e_c = -133;
    } else {
        e_c = exponent__5902d066__5ebbc1f4__ba49bfc41(c, -126);
    }
    float _tmp6 = es[0];
    for (size_t _tmp7 = 1; _tmp7 < es.size(); ++_tmp7) {
        auto&& _tmp8 = es[_tmp7];
        _tmp6 = ((std::isnan(_tmp6) || std::isnan(_tmp8)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp6 < _tmp8 || (_tmp6 == _tmp8 && std::signbit(_tmp6))) ? _tmp8 : _tmp6));
    }
    float e_max = ((std::isnan(_tmp6) || std::isnan(e_c)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp6 < e_c || (_tmp6 == e_c && std::signbit(_tmp6))) ? e_c : _tmp6));
    double s = fused_sum__5902d066__dcba4d50__v99d6777c__b81540742(join__5902d066__d0e61c18__b5e34ac9d(prods, std::array<float, 1>{{c}}), ((e_max - static_cast<float>(25)) - static_cast<float>(1)));
    return fdpa_round__5902d066__c58205ca__v5af88987__bc5f10b0e(s);
}
}
