"""
C++ backend: utilities
"""


CPP_HEADERS = [
    '#include <cassert>',
    '#include <cfenv>',
    '#include <cmath>',
    '#include <cstddef>',
    '#include <cstdint>',
    '#include <numeric>',
    '#include <vector>',
    '#include <tuple>',
]

CPP_HELPERS = """
template <typename T>
static size_t size(const T&, size_t) {
    assert(false && "cannot compute tensor size of a scalar");
    return 0;
}

template <typename T>
static size_t size(const std::vector<T>& vec, size_t n) {
    return (n == 0) ? vec.size() : size(vec[0], n);
}
"""
