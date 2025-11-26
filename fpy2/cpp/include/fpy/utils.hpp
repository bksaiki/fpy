#pragma once

#include <stdexcept>
#include <type_traits>

/// Static assertion
#define FPY_STATIC_ASSERT(cond, msg) static_assert(cond, msg);

/// Runtime assertion
#define FPY_ASSERT(cond, msg) \
    if (!(cond)) { \
        throw std::runtime_error("Assertion failed: " msg); \
    }

/// Unreachable assertion
#define FPY_UNREACHABLE(msg) \
    throw std::runtime_error("Unreachable code reached: " msg);

/// Debug assertion (only active in debug builds)
#ifdef FPY_DEBUG
    #define FPY_DEBUG_ASSERT(cond, msg) FPY_ASSERT(cond, msg)
#else
    #define FPY_DEBUG_ASSERT(cond, msg) static_assert(true);
#endif
