# Examples
add_executable(benchmark_add benchmark_add.cpp)
target_link_libraries(benchmark_add 
    fpy
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)
