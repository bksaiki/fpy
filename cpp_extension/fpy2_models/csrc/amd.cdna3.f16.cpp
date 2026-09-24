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

namespace fpy_models::model_amd_cdna3_f16 {

float overflow_inf__5902d066__c1f9c082__bc079c330(float p) {
    if ((static_cast<double>(std::fabs(p)) >= 3.402823669209385e+38)) {
        float t{};
        if (std::signbit(p)) {
            t = (-std::numeric_limits<float>::infinity());
        } else {
            t = std::numeric_limits<float>::infinity();
        }
        return t;
    }
    return p;
}

float sum_special_values__5902d066__ffcafc90__bad50af60(const std::array<float, 8>& ts, float c) {
    std::array<bool, 8> t7 = std::array<bool, 8>{};
    for (int8_t t8 = 0; t8 < 8; ++t8) {
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

float exponent__5902d066__67169c42__bf0928713(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

double fused_sum__5902d066__e46cb808__v99d6777c__b4a1aa3ca(const std::array<float, 8>& xs, float n) {
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

float exponent__5902d066__5ebbc1f4__ba49bfc41(float x, int8_t emin) {
    auto&& _tmp1 = std::logb(x);
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

double round_down_at__5902d066__4aa8fb08__b199cf636(double x, float n) {
    float _k = (n + static_cast<float>(1));
    auto&& _tmp1 = (-_k);
    double _t = (std::isfinite(_tmp1) ? std::ldexp(x, static_cast<int>(_tmp1)) : std::pow(2.0, _tmp1) * x);
    assert((std::isfinite(_t)) && "fpy: rounding is undefined for this value");
    double _tmp2 = std::floor(_t);
    assert((std::fabs(_tmp2) <= 274877906944) && "fpy: overflow occurred so rounding is undefined");
    double _t3 = _tmp2;
    double t = (std::isfinite(_k) ? std::ldexp(_t3, static_cast<int>(_k)) : std::pow(2.0, _k) * _t3);
    return t;
}

double round_down_at__5902d066__13da0761__b94d580d8(float x, float n) {
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

double round_down_at__5902d066__dd1c2a23__bfa2435e8(double x, float n) {
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

float exponent__5902d066__4bc6effc__b055c9940(double x, int8_t emin) {
    auto&& _tmp1 = static_cast<float>(std::logb(x));
    auto&& _tmp2 = static_cast<float>(emin);
    return ((std::isnan(_tmp1) || std::isnan(_tmp2)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp1 < _tmp2 || (_tmp1 == _tmp2 && std::signbit(_tmp1))) ? _tmp2 : _tmp1));
}

float tr_fdpa_block__5902d066__bf0555f2__bfa665f59(const std::array<float, 8>& A, const std::array<float, 8>& B, float c) {
    std::array<uint8_t, 8> _tmp1 = std::array<uint8_t, 8>{};
    std::iota(_tmp1.begin(), _tmp1.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 8> t37 = _tmp1;
    std::array<std::tuple<float, float>, 8> t27 = std::array<std::tuple<float, float>, 8>{};
    for (int8_t t38 = 0; t38 < 8; ++t38) {
        uint8_t t35 = t37[static_cast<size_t>(t38)];
        t27[static_cast<size_t>(t38)] = std::make_tuple(A[static_cast<size_t>(t35)], B[static_cast<size_t>(t35)]);
    }
    std::array<float, 8> prods = std::array<float, 8>{};
    for (int8_t t28 = 0; t28 < 8; ++t28) {
        auto&& _tmp2 = t27[static_cast<size_t>(t28)];
        float a = std::get<0>(_tmp2);
        float b = std::get<1>(_tmp2);
        prods[static_cast<size_t>(t28)] = overflow_inf__5902d066__c1f9c082__bc079c330((a * b));
    }
    std::array<bool, 8> t30 = std::array<bool, 8>{};
    for (int8_t t31 = 0; t31 < 8; ++t31) {
        float p = prods[static_cast<size_t>(t31)];
        t30[static_cast<size_t>(t31)] = (!std::isfinite(p));
    }
    bool t29 = std::any_of(t30.begin(), t30.end(), [](bool _tmp3) { return _tmp3; });
    if ((!t29)) {
        t29 = (!std::isfinite(c));
    }
    if (t29) {
        return sum_special_values__5902d066__ffcafc90__bad50af60(prods, c);
    }
    std::array<uint8_t, 8> _tmp4 = std::array<uint8_t, 8>{};
    std::iota(_tmp4.begin(), _tmp4.end(), static_cast<uint8_t>(0));
    std::array<uint8_t, 8> t39 = _tmp4;
    std::array<std::tuple<float, float, float>, 8> t32 = std::array<std::tuple<float, float, float>, 8>{};
    for (int8_t t40 = 0; t40 < 8; ++t40) {
        uint8_t t36 = t39[static_cast<size_t>(t40)];
        t32[static_cast<size_t>(t40)] = std::make_tuple(prods[static_cast<size_t>(t36)], A[static_cast<size_t>(t36)], B[static_cast<size_t>(t36)]);
    }
    std::array<float, 8> es = std::array<float, 8>{};
    for (int8_t t33 = 0; t33 < 8; ++t33) {
        auto&& _tmp5 = t32[static_cast<size_t>(t33)];
        float p_1 = std::get<0>(_tmp5);
        float a_1 = std::get<1>(_tmp5);
        float b_1 = std::get<2>(_tmp5);
        float t34{};
        if ((p_1 == static_cast<float>(0))) {
            t34 = -49;
        } else {
            t34 = (exponent__5902d066__67169c42__bf0928713(a_1, -14) + exponent__5902d066__67169c42__bf0928713(b_1, -14));
        }
        es[static_cast<size_t>(t33)] = t34;
    }
    float _tmp6 = es[0];
    for (size_t _tmp7 = 1; _tmp7 < es.size(); ++_tmp7) {
        auto&& _tmp8 = es[_tmp7];
        _tmp6 = ((std::isnan(_tmp6) || std::isnan(_tmp8)) ? std::numeric_limits<float>::quiet_NaN() : ((_tmp6 < _tmp8 || (_tmp6 == _tmp8 && std::signbit(_tmp6))) ? _tmp8 : _tmp6));
    }
    float e_dot = _tmp6;
    double t = fused_sum__5902d066__e46cb808__v99d6777c__b4a1aa3ca(prods, ((e_dot - static_cast<float>(24)) - static_cast<float>(1)));
    float e_c = exponent__5902d066__5ebbc1f4__ba49bfc41(c, -126);
    float e = ((std::isnan(e_dot) || std::isnan(e_c)) ? std::numeric_limits<float>::quiet_NaN() : ((e_dot < e_c || (e_dot == e_c && std::signbit(e_dot))) ? e_c : e_dot));
    double t_1 = round_down_at__5902d066__4aa8fb08__b199cf636(t, ((e - static_cast<float>(31)) - static_cast<float>(2)));
    double cr = round_down_at__5902d066__13da0761__b94d580d8(c, ((e - static_cast<float>(24)) - static_cast<float>(1)));
    double s = (t_1 + cr);
    double s_1 = round_down_at__5902d066__dd1c2a23__bfa2435e8(s, ((exponent__5902d066__4bc6effc__b055c9940(s, -126) - static_cast<float>(31)) - static_cast<float>(1)));
    return static_cast<float>(s_1);
}

float tr_fdpa(const std::array<float, 8>& A, const std::array<float, 8>& B, float c) {
    float d = c;
    for (int8_t i = 0; i < 8; i += 8) {
        std::array<float, 8> _tmp1{};
        std::copy(A.begin() + static_cast<size_t>(i), A.begin() + static_cast<size_t>((i + 8)), _tmp1.begin());
        std::array<float, 8> _tmp2{};
        std::copy(B.begin() + static_cast<size_t>(i), B.begin() + static_cast<size_t>((i + 8)), _tmp2.begin());
        d = tr_fdpa_block__5902d066__bf0555f2__bfa665f59(_tmp1, _tmp2, d);
    }
    return d;
}
}
