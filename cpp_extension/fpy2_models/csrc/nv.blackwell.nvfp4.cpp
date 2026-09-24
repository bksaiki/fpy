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

namespace fpy_models::model_nv_blackwell_nvfp4 {

float dpa_special_values__5902d066__8a9d0219__b5ed31f40(const std::array<float, 64>& A, const std::array<float, 64>& B, float c) {
    std::array<bool, 64> t10 = std::array<bool, 64>{};
    for (int8_t t11 = 0; t11 < 64; ++t11) {
        float a = A[static_cast<size_t>(t11)];
        t10[static_cast<size_t>(t11)] = std::isnan(a);
    }
    std::array<bool, 64> t13 = std::array<bool, 64>{};
    for (int8_t t14 = 0; t14 < 64; ++t14) {
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
    for (int8_t _i = 0; _i < 64; ++_i) {
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

float exponent__5902d066__6b2128aa__b3991cffa(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

float exponent__5902d066__5ebbc1f4__ba49bfc41(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

double fused_sum__5902d066__dbc5b73e__v99d6777c__be4e9a145(const std::array<float, 5>& xs, float n) {
    std::array<double, 5> ts = std::array<double, 5>{};
    for (int8_t t6 = 0; t6 < 5; ++t6) {
        float x = xs[static_cast<size_t>(t6)];
        float _k = (n + static_cast<float>(1));
        auto&& _tmp1 = (-_k);
        auto&& _tmp2 = static_cast<double>(x);
        double _t = (std::isfinite(_tmp1) ? std::ldexp(_tmp2, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * _tmp2);
        assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
        float _tmp3 = std::trunc(_t);
        assert((std::fabs(_tmp3) <= 140737488355328) && "fpy: overflow occurred so rounding is undefined");
        float _t9 = _tmp3;
        auto&& _tmp4 = static_cast<double>(_t9);
        double t8 = (std::isfinite(_k) ? std::ldexp(_tmp4, static_cast<int>(_k)) : std::pow(2.0, _k) * _tmp4);
        ts[static_cast<size_t>(t6)] = t8;
    }
    return std::accumulate(ts.begin() + 1, ts.end(), ts[static_cast<size_t>(0)]);
}

std::array<float, 5> join__5902d066__c8eb9f92__b3ff51ed7(const std::array<float, 4>& xs, const std::array<float, 1>& ys) {
    std::array<float, 5> zs = std::array<float, 5>{};
    for (int8_t i = 0; i < 4; ++i) {
        zs[static_cast<size_t>(i)] = xs[static_cast<size_t>(i)];
    }
    for (int8_t i_1 = 0; i_1 < 1; ++i_1) {
        zs[static_cast<size_t>((4 + i_1))] = ys[static_cast<size_t>(i_1)];
    }
    return zs;
}

float fdpa_round__5902d066__d89f5181__v5af88987__bf2bc6419(double s) {
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

float gst_fdpa(const std::array<float, 64>& A, const std::array<float, 64>& B, float c, const std::array<float, 4>& alphas, const std::array<float, 4>& betas) {
    std::array<bool, 4> t32 = std::array<bool, 4>{};
    for (int8_t t33 = 0; t33 < 4; ++t33) {
        float s1 = alphas[static_cast<size_t>(t33)];
        t32[static_cast<size_t>(t33)] = std::isnan(s1);
    }
    std::array<bool, 4> t35 = std::array<bool, 4>{};
    for (int8_t t36 = 0; t36 < 4; ++t36) {
        float s2 = betas[static_cast<size_t>(t36)];
        t35[static_cast<size_t>(t36)] = std::isnan(s2);
    }
    bool t = std::any_of(t32.begin(), t32.end(), [](bool _tmp1) { return _tmp1; });
    if ((!t)) {
        t = std::any_of(t35.begin(), t35.end(), [](bool _tmp2) { return _tmp2; });
    }
    if (t) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::array<int8_t, 64> _tmp3 = std::array<int8_t, 64>{};
    std::iota(_tmp3.begin(), _tmp3.end(), static_cast<int8_t>(0));
    std::array<int8_t, 64> t46 = _tmp3;
    std::array<std::tuple<float, float>, 64> t37 = std::array<std::tuple<float, float>, 64>{};
    for (int8_t t47 = 0; t47 < 64; ++t47) {
        int8_t t45 = t46[static_cast<size_t>(t47)];
        t37[static_cast<size_t>(t47)] = std::make_tuple(A[static_cast<size_t>(t45)], B[static_cast<size_t>(t45)]);
    }
    std::array<float, 64> prods = std::array<float, 64>{};
    for (int8_t t38 = 0; t38 < 64; ++t38) {
        auto&& _tmp4 = t37[static_cast<size_t>(t38)];
        float a = std::get<0>(_tmp4);
        float b = std::get<1>(_tmp4);
        prods[static_cast<size_t>(t38)] = (a * b);
    }
    std::array<bool, 64> t40 = std::array<bool, 64>{};
    for (int8_t t41 = 0; t41 < 64; ++t41) {
        float p = prods[static_cast<size_t>(t41)];
        t40[static_cast<size_t>(t41)] = (!std::isfinite(p));
    }
    bool t39 = std::any_of(t40.begin(), t40.end(), [](bool _tmp5) { return _tmp5; });
    if ((!t39)) {
        t39 = (!std::isfinite(c));
    }
    if (t39) {
        return dpa_special_values__5902d066__8a9d0219__b5ed31f40(A, B, c);
    }
    std::array<float, 4> ts = std::array<float, 4>{};
    std::array<float, 4> es = std::array<float, 4>{};
    for (int8_t g = 0; g < 4; ++g) {
        std::array<float, 16> _tmp6{};
        std::copy(prods.begin() + static_cast<size_t>((g * 16)), prods.begin() + static_cast<size_t>(((g + 1) * 16)), _tmp6.begin());
        std::array<float, 16> group = _tmp6;
        float scale = (alphas[static_cast<size_t>(g)] * betas[static_cast<size_t>(g)]);
        ts[static_cast<size_t>(g)] = (std::accumulate(group.begin() + 1, group.end(), group[static_cast<size_t>(0)]) * scale);
        std::array<bool, 16> t43 = std::array<bool, 16>{};
        for (int8_t t44 = 0; t44 < 16; ++t44) {
            float p_1 = group[static_cast<size_t>(t44)];
            t43[static_cast<size_t>(t44)] = (p_1 == static_cast<float>(0));
        }
        bool t42 = std::all_of(t43.begin(), t43.end(), [](bool _tmp7) { return _tmp7; });
        if ((!t42)) {
            t42 = (scale == static_cast<float>(0));
        }
        if (t42) {
            es[static_cast<size_t>(g)] = -139;
        } else {
            es[static_cast<size_t>(g)] = (exponent__5902d066__6b2128aa__b3991cffa(alphas[static_cast<size_t>(g)], -6) + exponent__5902d066__6b2128aa__b3991cffa(betas[static_cast<size_t>(g)], -6));
        }
    }
    float e_c{};
    if ((c == static_cast<float>(0))) {
        e_c = -139;
    } else {
        e_c = exponent__5902d066__5ebbc1f4__ba49bfc41(c, -126);
    }
    float _tmp8 = es[0];
    for (size_t _tmp9 = 1; _tmp9 < es.size(); ++_tmp9) {
        auto&& _tmp10 = es[_tmp9];
        _tmp8 = ((std::isnan(_tmp8) || std::isnan(_tmp10)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp8 < _tmp10 || (_tmp8 == _tmp10 && std::signbit(_tmp8))) ? _tmp10 : _tmp8));
    }
    float e_max = ((std::isnan(_tmp8) || std::isnan(e_c)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp8 < e_c || (_tmp8 == e_c && std::signbit(_tmp8))) ? e_c : _tmp8));
    double s = fused_sum__5902d066__dbc5b73e__v99d6777c__be4e9a145(join__5902d066__c8eb9f92__b3ff51ed7(ts, std::array<float, 1>{{c}}), ((e_max - static_cast<float>(35)) - static_cast<float>(1)));
    return fdpa_round__5902d066__d89f5181__v5af88987__bf2bc6419(s);
}
}
