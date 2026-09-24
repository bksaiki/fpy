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

namespace fpy_models::model_amd_cdna3_bf8 {

float exponent__5902d066__1a566fa7__b33b80d31(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

float exponent0__5902d066__1a566fa7__ba4025fd4(float x, int8_t emin) {
    if ((!std::isfinite(x))) {
        return -1;
    }
    return exponent__5902d066__1a566fa7__b33b80d31(x, emin);
}

float sum_special_values__5902d066__f5e44fbd__bfd85773d(const std::array<float, 16>& ts, float c) {
    std::array<bool, 16> t7 = std::array<bool, 16>{};
    for (int8_t t8 = 0; t8 < 16; ++t8) {
        float t = ts[static_cast<size_t>(t8)];
        t7[static_cast<size_t>(t8)] = std::isnan(t);
    }
    bool t9 = std::any_of(t7.begin(), t7.end(), [](bool _tmp1) { return _tmp1; });
    if ((!t9)) {
        t9 = std::isnan(c);
    }
    if (t9) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    bool has_inf = std::isinf(c);
    bool inf_sgn = std::signbit(c);
    for (float t_1 : ts) {
        if (std::isinf(t_1)) {
            bool t_sgn = std::signbit(t_1);
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
    float t10{};
    if (inf_sgn) {
        t10 = (-std::numeric_limits<float>::infinity());
    } else {
        t10 = std::numeric_limits<float>::infinity();
    }
    return t10;
}

double fused_sum__5902d066__5e7a541e__v99d6777c__bf08fdbfc(const std::array<float, 8>& xs, float n) {
    std::array<float, 8> ts = std::array<float, 8>{};
    for (int8_t t6 = 0; t6 < 8; ++t6) {
        float x = xs[static_cast<size_t>(t6)];
        float _k = (n + static_cast<float>(1));
        auto&& _tmp1 = (-_k);
        float _t = (std::isfinite(_tmp1) ? std::ldexp(x, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * x);
        assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
        float _tmp2 = std::trunc(_t);
        assert((std::fabs(_tmp2) <= 67108864) && "fpy: overflow occurred so rounding is undefined");
        float _t9 = _tmp2;
        float t8 = (std::isfinite(_k) ? std::ldexp(_t9, static_cast<int>(_k)) : std::pow(2.0, _k) * _t9);
        ts[static_cast<size_t>(t6)] = t8;
    }
    return std::accumulate(ts.begin() + 1, ts.end(), static_cast<double>(ts[static_cast<size_t>(0)]));
}

double round_down_at__5902d066__410f0bc0__b885e9e71(double x, float n) {
    float _k = (n + static_cast<float>(1));
    auto&& _tmp1 = (-_k);
    double _t = (std::isfinite(_tmp1) ? std::ldexp(x, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * x);
    assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
    double _tmp2 = std::floor(_t);
    assert((std::fabs(_tmp2) <= 1073741824) && "fpy: overflow occurred so rounding is undefined");
    double _t3 = _tmp2;
    double t = (std::isfinite(_k) ? std::ldexp(_t3, static_cast<int>(_k)) : std::pow(2.0, _k) * _t3);
    return t;
}

float exponent__5902d066__5ebbc1f4__ba49bfc41(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

double round_down_at__5902d066__9054860a__b7d82751b(double x, float n) {
    float _k = (n + static_cast<float>(1));
    auto&& _tmp1 = (-_k);
    double _t = (std::isfinite(_tmp1) ? std::ldexp(x, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * x);
    assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
    double _tmp2 = std::floor(_t);
    assert((std::fabs(_tmp2) <= 1099511627776) && "fpy: overflow occurred so rounding is undefined");
    double _t3 = _tmp2;
    double t = (std::isfinite(_k) ? std::ldexp(_t3, static_cast<int>(_k)) : std::pow(2.0, _k) * _t3);
    return t;
}

double round_down_at__5902d066__830edead__bb0906bf2(float x, float n) {
    float _k = (n + static_cast<float>(1));
    auto&& _tmp1 = (-_k);
    auto&& _tmp2 = static_cast<double>(x);
    double _t = (std::isfinite(_tmp1) ? std::ldexp(_tmp2, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * _tmp2);
    assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
    float _tmp3 = std::floor(_t);
    assert((std::fabs(_tmp3) <= 67108864) && "fpy: overflow occurred so rounding is undefined");
    float _t3 = _tmp3;
    auto&& _tmp4 = static_cast<double>(_t3);
    double t = (std::isfinite(_k) ? std::ldexp(_tmp4, static_cast<int>(_k)) : std::pow(2.0, _k) * _tmp4);
    return t;
}

double round_down_at__5902d066__6e0af161__bcd39e169(double x, float n) {
    float _k = (n + static_cast<float>(1));
    auto&& _tmp1 = (-_k);
    double _t = (std::isfinite(_tmp1) ? std::ldexp(x, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * x);
    assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
    double _tmp2 = std::floor(_t);
    assert((std::fabs(_tmp2) <= 8589934592) && "fpy: overflow occurred so rounding is undefined");
    double _t3 = _tmp2;
    double t = (std::isfinite(_k) ? std::ldexp(_t3, static_cast<int>(_k)) : std::pow(2.0, _k) * _t3);
    return t;
}

float exponent__5902d066__9c81a4c2__b69691637(double x, int8_t emin) {
    auto&& _tmp1 = static_cast<float>(std::logb(x));
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

float gtr_fdpa_block__5902d066__5730feff__b04928e61(const std::array<float, 16>& A, const std::array<float, 16>& B, float c) {
    std::array<uint8_t, 16> _tmp1 = std::array<uint8_t, 16>{};
    std::iota(_tmp1.begin(), _tmp1.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 16> t52 = _tmp1;
    std::array<std::tuple<float, float>, 16> t33 = std::array<std::tuple<float, float>, 16>{};
    for (int8_t t53 = 0; t53 < 16; ++t53) {
        uint8_t t50 = t52[static_cast<size_t>(t53)];
        t33[static_cast<size_t>(t53)] = std::make_tuple(A[static_cast<size_t>(t50)], B[static_cast<size_t>(t50)]);
    }
    std::array<float, 16> prods = std::array<float, 16>{};
    for (int8_t t34 = 0; t34 < 16; ++t34) {
        auto&& _tmp2 = t33[static_cast<size_t>(t34)];
        float a = std::get<0>(_tmp2);
        float b = std::get<1>(_tmp2);
        prods[static_cast<size_t>(t34)] = (a * b);
    }
    std::array<uint8_t, 16> _tmp3 = std::array<uint8_t, 16>{};
    std::iota(_tmp3.begin(), _tmp3.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 16> t54 = _tmp3;
    std::array<std::tuple<float, float, float>, 16> t35 = std::array<std::tuple<float, float, float>, 16>{};
    for (int8_t t55 = 0; t55 < 16; ++t55) {
        uint8_t t51 = t54[static_cast<size_t>(t55)];
        t35[static_cast<size_t>(t55)] = std::make_tuple(prods[static_cast<size_t>(t51)], A[static_cast<size_t>(t51)], B[static_cast<size_t>(t51)]);
    }
    std::array<float, 16> es = std::array<float, 16>{};
    for (int8_t t36 = 0; t36 < 16; ++t36) {
        auto&& _tmp4 = t35[static_cast<size_t>(t36)];
        float p = std::get<0>(_tmp4);
        float a_1 = std::get<1>(_tmp4);
        float b_1 = std::get<2>(_tmp4);
        float t41{};
        if ((p == static_cast<float>(0))) {
            t41 = -35;
        } else {
            t41 = (exponent0__5902d066__1a566fa7__ba4025fd4(a_1, -15) + exponent0__5902d066__1a566fa7__ba4025fd4(b_1, -15));
        }
        es[static_cast<size_t>(t36)] = t41;
    }
    std::array<float, 8> t37 = std::array<float, 8>{};
    uint8_t t38 = 0;
    for (int8_t i = 0; i < 16; i += 2) {
        t37[static_cast<size_t>(t38)] = es[static_cast<size_t>(i)];
        uint8_t _t50 = (t38 + 1);
        t38 = _t50;
    }
    float _tmp5 = t37[0];
    for (size_t _tmp6 = 1; _tmp6 < t37.size(); ++_tmp6) {
        auto&& _tmp7 = t37[_tmp6];
        _tmp5 = ((std::isnan(_tmp5) || std::isnan(_tmp7)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp5 < _tmp7 || (_tmp5 == _tmp7 && std::signbit(_tmp5))) ? _tmp7 : _tmp5));
    }
    float e_even = _tmp5;
    std::array<float, 8> t39 = std::array<float, 8>{};
    uint8_t t40 = 0;
    for (int8_t i_1 = 1; i_1 < 16; i_1 += 2) {
        t39[static_cast<size_t>(t40)] = es[static_cast<size_t>(i_1)];
        uint8_t _t52 = (t40 + 1);
        t40 = _t52;
    }
    float _tmp8 = t39[0];
    for (size_t _tmp9 = 1; _tmp9 < t39.size(); ++_tmp9) {
        auto&& _tmp10 = t39[_tmp9];
        _tmp8 = ((std::isnan(_tmp8) || std::isnan(_tmp10)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp8 < _tmp10 || (_tmp8 == _tmp10 && std::signbit(_tmp8))) ? _tmp10 : _tmp8));
    }
    float e_odd = _tmp8;
    bool t48 = (!std::isfinite(c));
    if (t48) {
        t48 = (((std::isnan(e_even) || std::isnan(e_odd)) ? std::numeric_limits<float>::quiet_NaN() : ((e_even < e_odd || (e_even == e_odd && std::signbit(e_even))) ? e_odd : e_even)) > static_cast<float>(24));
    }
    if (t48) {
        c = 0;
    }
    std::array<bool, 16> t42 = std::array<bool, 16>{};
    for (int8_t t43 = 0; t43 < 16; ++t43) {
        float p_1 = prods[static_cast<size_t>(t43)];
        t42[static_cast<size_t>(t43)] = (!std::isfinite(p_1));
    }
    bool t49 = std::any_of(t42.begin(), t42.end(), [](bool _tmp11) { return _tmp11; });
    if ((!t49)) {
        t49 = (!std::isfinite(c));
    }
    if (t49) {
        return sum_special_values__5902d066__f5e44fbd__bfd85773d(prods, c);
    }
    std::array<float, 8> t44 = std::array<float, 8>{};
    uint8_t t45 = 0;
    for (int8_t i_2 = 0; i_2 < 16; i_2 += 2) {
        t44[static_cast<size_t>(t45)] = prods[static_cast<size_t>(i_2)];
        uint8_t _t54 = (t45 + 1);
        t45 = _t54;
    }
    double t_even = fused_sum__5902d066__5e7a541e__v99d6777c__bf08fdbfc(t44, ((e_even - static_cast<float>(24)) - static_cast<float>(1)));
    std::array<float, 8> t46 = std::array<float, 8>{};
    uint8_t t47 = 0;
    for (int8_t i_3 = 1; i_3 < 16; i_3 += 2) {
        t46[static_cast<size_t>(t47)] = prods[static_cast<size_t>(i_3)];
        uint8_t _t56 = (t47 + 1);
        t47 = _t56;
    }
    double t_odd = fused_sum__5902d066__5e7a541e__v99d6777c__bf08fdbfc(t46, ((e_odd - static_cast<float>(24)) - static_cast<float>(1)));
    float e_dot = ((std::isnan(e_even) || std::isnan(e_odd)) ? std::numeric_limits<float>::quiet_NaN() : ((e_even < e_odd || (e_even == e_odd && std::signbit(e_even))) ? e_odd : e_even));
    double t = (round_down_at__5902d066__410f0bc0__b885e9e71(t_even, ((e_dot - static_cast<float>(24)) - static_cast<float>(1))) + round_down_at__5902d066__410f0bc0__b885e9e71(t_odd, ((e_dot - static_cast<float>(24)) - static_cast<float>(1))));
    float e_c = exponent__5902d066__5ebbc1f4__ba49bfc41(c, -126);
    float e = ((std::isnan(e_dot) || std::isnan(e_c)) ? std::numeric_limits<float>::quiet_NaN() : ((e_dot < e_c || (e_dot == e_c && std::signbit(e_dot))) ? e_c : e_dot));
    double t_1 = round_down_at__5902d066__9054860a__b7d82751b(t, ((e - static_cast<float>(31)) - static_cast<float>(2)));
    double cr{};
    if ((e_c < ((e - static_cast<float>(24)) - static_cast<float>(1)))) {
        cr = 0;
    } else {
        cr = round_down_at__5902d066__830edead__bb0906bf2(c, ((e - static_cast<float>(24)) - static_cast<float>(1)));
    }
    double s = (t_1 + cr);
    double s_1 = round_down_at__5902d066__6e0af161__bcd39e169(s, ((exponent__5902d066__9c81a4c2__b69691637(s, -126) - static_cast<float>(31)) - static_cast<float>(1)));
    return static_cast<float>(s_1);
}

float gtr_fdpa(const std::array<float, 16>& A, const std::array<float, 16>& B, float c) {
    float d = c;
    for (int8_t i = 0; i < 16; i += 16) {
        std::array<float, 16> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 16)), _tmp1.begin());
        std::array<float, 16> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 16)), _tmp2.begin());
        d = gtr_fdpa_block__5902d066__5730feff__b04928e61(_tmp1, _tmp2, d);
    }
    return d;
}
}
