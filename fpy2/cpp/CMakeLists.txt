cmake_minimum_required(VERSION 3.15)

# NOTE: Update project name, version, and description as needed
project(fpy_numbers
    VERSION 0.1.0
    DESCRIPTION "Number library for software floating-point arithmetic"
    LANGUAGES CXX
)

# C++ standard (GoogleTest requires C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)

# Add GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0 # Use a specific, stable release tag
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# NOTE: Add external dependencies here (e.g., GMP, MPFR if needed)
# Example for GMP/MPFR:
# find_package(GMP REQUIRED)
# find_package(MPFR REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    # NOTE: Add additional include directories here
)

# Source files
# NOTE: Add your source files here as you implement them
set(SOURCES
    src/rounding.cpp
)

# Header files (for IDE organization)
set(HEADERS
    include/fpy/rounding.hpp
    include/fpy/types.hpp
)

# Create library
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# NOTE: Link external libraries here
# Example for GMP/MPFR:
# target_link_libraries(${PROJECT_NAME} PUBLIC gmp mpfr)

# Set library properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${HEADERS}"
)

# Install targets
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/fpy
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# NOTE: Consider adding:
# - Documentation generation (Doxygen)
# - Code coverage options
# - Sanitizer options for debugging
# - Package config files for easy integration
