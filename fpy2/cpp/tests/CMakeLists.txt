# Tests CMakeLists.txt

# Test executables
set(TEST_SOURCES
    test_rounding.cpp
)

# Create test executable
add_executable(run_tests ${TEST_SOURCES})

# Link with main library and testing framework
target_link_libraries(run_tests
    fpy_numbers
    GTest::gtest_main
)

# Register tests with CTest
add_test(NAME AllTests COMMAND run_tests)
