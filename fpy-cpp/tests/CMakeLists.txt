# Tests CMakeLists.txt

# Test executables
set(TEST_SOURCES
    test_context_mp.cpp
    test_ops.cpp
    test_real.cpp
    test_round.cpp
    test_round_opt.cpp
)

# Create test executable
add_executable(run_tests ${TEST_SOURCES})

# Link with main library and testing framework
target_link_libraries(run_tests
    fpy
    GTest::gtest_main
    ${GMP_LIBRARY}
    ${MPFR_LIBRARY}
)

# Register tests with CTest
add_test(NAME AllTests COMMAND run_tests)
