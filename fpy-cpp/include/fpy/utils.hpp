#pragma once

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <string>

/// Helper macro to stringify a value
#define FPY_STRINGIFY(x) #x
#define FPY_TOSTRING(x) FPY_STRINGIFY(x)

/// Static assertion
#define FPY_STATIC_ASSERT(cond, msg) static_assert(cond, msg)

/// Runtime assertion
#define FPY_ASSERT(cond, msg) \
    if (!(cond)) { \
        throw std::runtime_error( \
            std::string("Assertion failed at ") + __FILE__ + ":" + \
            FPY_TOSTRING(__LINE__) + ": " + msg); \
    }

/// Unreachable assertion
#define FPY_UNREACHABLE(...) \
    throw std::runtime_error( \
        std::string("Unreachable code reached at ") + __FILE__ + ":" + \
        FPY_TOSTRING(__LINE__) + \
        (sizeof(#__VA_ARGS__) > 1 ? (": " + std::string(__VA_ARGS__)) : "") \
    );

/// Debug assertion (only active in debug builds)
#ifdef FPY_DEBUG
    #define FPY_DEBUG_ASSERT(cond, msg) FPY_ASSERT(cond, msg)
#else
    #define FPY_DEBUG_ASSERT(cond, msg) ((void) (cond))
#endif

// Branch prediction hints
#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif
