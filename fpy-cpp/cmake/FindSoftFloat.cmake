# Find SoftFloat library
# Sets:
#   SOFTFLOAT_FOUND - True if SoftFloat is found
#   SOFTFLOAT_INCLUDE_DIRS - Include directories
#   SOFTFLOAT_LIBRARIES - Libraries to link against

find_path(SOFTFLOAT_INCLUDE_DIR
    NAMES softfloat.h
    PATHS
        ${CMAKE_SOURCE_DIR}/third_party/softfloat/include
        ${CMAKE_SOURCE_DIR}/../third_party/softfloat/include
        /usr/include
        /usr/local/include
        /opt/local/include
)

find_library(SOFTFLOAT_LIBRARY
    NAMES softfloat libsoftfloat
    PATHS
        ${CMAKE_SOURCE_DIR}/third_party/softfloat/lib
        ${CMAKE_SOURCE_DIR}/../third_party/softfloat/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SoftFloat
    REQUIRED_VARS SOFTFLOAT_LIBRARY SOFTFLOAT_INCLUDE_DIR
)

if(SOFTFLOAT_FOUND)
    set(SOFTFLOAT_INCLUDE_DIRS ${SOFTFLOAT_INCLUDE_DIR})
    set(SOFTFLOAT_LIBRARIES ${SOFTFLOAT_LIBRARY})
    
    # Create imported target
    if(NOT TARGET SoftFloat::softfloat)
        add_library(SoftFloat::softfloat STATIC IMPORTED)
        set_target_properties(SoftFloat::softfloat PROPERTIES
            IMPORTED_LOCATION ${SOFTFLOAT_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${SOFTFLOAT_INCLUDE_DIR}
        )
    endif()
endif()

mark_as_advanced(SOFTFLOAT_INCLUDE_DIR SOFTFLOAT_LIBRARY)
