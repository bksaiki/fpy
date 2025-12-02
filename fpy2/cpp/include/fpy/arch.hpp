#pragma once

#include <cfenv>

// Architecture-specific includes and definitions
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <xmmintrin.h>
    #define FPY_ARCH_X86
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define FPY_ARCH_ARM64
#elif defined(__arm__) || defined(_M_ARM)
    #define FPY_ARCH_ARM32
#else
    #define FPY_ARCH_GENERIC
#endif

namespace fpy {
namespace arch {

#ifdef FPY_ARCH_X86
    // x86/x64 implementation using SSE
    inline unsigned int get_fpscr() {
        return _mm_getcsr();
    }

    inline void set_fpscr(unsigned int csr) {
        _mm_setcsr(csr);
    }
    
    inline void clear_exceptions() {
        unsigned int csr = get_fpscr();
        set_fpscr(csr & ~0x3F); // Clear exception flags (bits 0-5)
    }

    inline bool has_exception(unsigned int flags) {
        return (get_fpscr() & flags) != 0;
    }

    inline int get_rounding_mode() {
        return (get_fpscr() >> 13) & 0x3; // Extract bits 13-14
    }
    
    inline void set_rounding_mode(int mode) {
        unsigned int csr = get_fpscr();
        csr = (csr & ~0x6000) | ((mode & 0x3) << 13); // Clear and set bits 13-14
        set_fpscr(csr);
    }
    
    inline int set_rtz() {
        const int old_mode = get_rounding_mode();
        set_rounding_mode(0x3); // RTZ mode for x86 SSE
        return old_mode;
    }
    
    // High-level operation management functions
    inline int prepare_rto() {
        unsigned int csr = get_fpscr();
        const int old_mode = (csr >> 13) & 0x3;  // Extract old rounding mode
        csr = (csr & ~0x6000) | (0x3 << 13);     // Set RTZ mode
        csr &= ~0x3F;                            // Clear exception flags
        set_fpscr(csr);
        return old_mode;
    }

    inline int rto_status(int old_mode) {
        unsigned int csr = get_fpscr();
        const int exceptions = csr & 0x38;       // Extract overflow, underflow, inexact flags
        csr = (csr & ~0x6000) | ((old_mode & 0x3) << 13);  // Restore rounding mode
        set_fpscr(csr);
        return exceptions;
    }
    
    // Exception flag constants
    static constexpr unsigned int EXCEPT_INVALID = 0x01;
    static constexpr unsigned int EXCEPT_DENORM = 0x02;
    static constexpr unsigned int EXCEPT_DIVZERO = 0x04;
    static constexpr unsigned int EXCEPT_OVERFLOW = 0x08;
    static constexpr unsigned int EXCEPT_UNDERFLOW = 0x10;
    static constexpr unsigned int EXCEPT_INEXACT = 0x20;
    
#elif defined(FPY_ARCH_ARM64)
    // ARM64 implementation using system registers
    inline unsigned int get_fpscr() {
        unsigned int fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        return fpcr;
    }
    
    inline void set_fpscr(unsigned int csr) {
        __asm__ volatile("msr fpcr, %0" : : "r"(csr));
    }
    
    inline void clear_exceptions() {
        unsigned int fpsr = 0;
        __asm__ volatile("msr fpsr, %0" : : "r"(fpsr));
    }

    inline bool has_exception(unsigned int flags) {
        unsigned int fpsr;
        __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr));
        return (fpsr & flags) != 0;
    }

    inline int get_rounding_mode() {
        return (get_fpscr() >> 22) & 0x3; // Extract RMode bits 23-22
    }
    
    inline void set_rounding_mode(int mode) {
        unsigned int fpcr = get_fpscr();
        fpcr = (fpcr & ~0xC00000) | ((mode & 0x3) << 22); // Clear and set bits 23-22
        set_fpscr(fpcr);
    }
    
    inline int set_rtz() {
        const int old_mode = get_rounding_mode();
        set_rounding_mode(0x3); // RTZ mode for ARM64
        return old_mode;
    }
    
    // High-level operation management functions
    inline int prepare_rto() {
        unsigned int fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        const int old_mode = (fpcr >> 22) & 0x3;  // Extract old rounding mode
        fpcr = (fpcr & ~0xC00000) | (0x3 << 22);  // Set RTZ mode
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
        __asm__ volatile("msr fpsr, %0" : : "r"(0)); // Clear exceptions
        return old_mode;
    }

    inline int rto_status(int old_mode) {
        unsigned int fpsr, fpcr;
        __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr));
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        const int exceptions = fpsr & 0x1E;       // Extract overflow, underflow, inexact flags
        fpcr = (fpcr & ~0xC00000) | ((old_mode & 0x3) << 22);  // Restore rounding mode
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
        return exceptions;
    }
    
    // Exception flag constants for ARM64
    static constexpr unsigned int EXCEPT_INVALID = 0x01;
    static constexpr unsigned int EXCEPT_DIVZERO = 0x02;
    static constexpr unsigned int EXCEPT_OVERFLOW = 0x04;
    static constexpr unsigned int EXCEPT_UNDERFLOW = 0x08;
    static constexpr unsigned int EXCEPT_INEXACT = 0x10;
    
#else
    // Generic implementation using standard C library
    inline int get_fpscr() {
        return std::fegetround();
    }
    
    inline void set_fpscr(int rm) {
        std::fesetround(rm);
    }
    
    inline void clear_exceptions() {
        std::feclearexcept(FE_ALL_EXCEPT);
    }

    inline bool has_exception(int flags) {
        return std::fetestexcept(flags) != 0;
    }

    // Rounding mode functions for generic
    inline int get_rounding_mode() {
        return std::fegetround();
    }
    
    inline void set_rounding_mode(int mode) {
        std::fesetround(mode);
    }
    
    inline int set_rtz() {
        const int old_mode = get_rounding_mode();
        set_rounding_mode(FE_TOWARDZERO);
        return old_mode;
    }
    
    // High-level operation management functions
    inline int prepare_rto() {
        const int old_mode = std::fegetround();
        std::fesetround(FE_TOWARDZERO);
        std::feclearexcept(FE_ALL_EXCEPT);
        return old_mode;
    }

    inline int rto_status(int old_mode) {
        const int exceptions = std::fetestexcept(FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
        std::fesetround(old_mode);
        return exceptions;
    }
    
    // Use standard exception constants
    static constexpr int EXCEPT_INVALID = FE_INVALID;
    static constexpr int EXCEPT_DIVZERO = FE_DIVBYZERO;
    static constexpr int EXCEPT_OVERFLOW = FE_OVERFLOW;
    static constexpr int EXCEPT_UNDERFLOW = FE_UNDERFLOW;
    static constexpr int EXCEPT_INEXACT = FE_INEXACT;
#endif

} // end namespace arch
} // end namespace fpy
