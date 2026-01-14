# Examples

add_executable(benchmark_add benchmark_add.cpp)
target_link_libraries(benchmark_add 
    fpy
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)
if(USE_SOFTFLOAT AND SOFTFLOAT_FOUND)
    target_link_libraries(benchmark_add SoftFloat::softfloat)
endif()

add_executable(benchmark_mul benchmark_mul.cpp)
target_link_libraries(benchmark_mul 
    fpy
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)
if(USE_SOFTFLOAT AND SOFTFLOAT_FOUND)
    target_link_libraries(benchmark_mul SoftFloat::softfloat)
endif()

add_executable(benchmark_div benchmark_div.cpp)
target_link_libraries(benchmark_div 
    fpy
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)
if(USE_SOFTFLOAT AND SOFTFLOAT_FOUND)
    target_link_libraries(benchmark_div SoftFloat::softfloat)
endif()

add_executable(benchmark_fma benchmark_fma.cpp)
target_link_libraries(benchmark_fma 
    fpy
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)
if(USE_SOFTFLOAT AND SOFTFLOAT_FOUND)
    target_link_libraries(benchmark_fma SoftFloat::softfloat)
endif()

add_executable(benchmark_sqrt benchmark_sqrt.cpp)
target_link_libraries(benchmark_sqrt 
    fpy
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)
if(USE_SOFTFLOAT AND SOFTFLOAT_FOUND)
    target_link_libraries(benchmark_sqrt SoftFloat::softfloat)
endif()

add_executable(benchmark_mul_engines benchmark_mul_engines.cpp)
target_link_libraries(benchmark_mul_engines fpy)

add_executable(mixed_dot_prod mixed_dot_prod.cpp)
target_link_libraries(mixed_dot_prod fpy)
