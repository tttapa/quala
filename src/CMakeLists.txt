# Target options
# --------------

add_library(quala-obj OBJECT
    "src/lbfgs.cpp"

    "include/quala/anderson-acceleration.hpp"
    "include/quala/broyden-good.hpp"
    "include/quala/lbfgs.hpp"
    "include/quala/decl/lbfgs.hpp"
    "include/quala/decl/lbfgs-fwd.hpp"
    "include/quala/detail/limited-memory-qr.hpp"
    "include/quala/detail/anderson-helpers.hpp"
    "include/quala/util/alloc.hpp"
    "include/quala/util/ringbuffer.hpp"
    "include/quala/util/vec.hpp"
)
add_library(quala::quala-obj ALIAS quala-obj)
set_property(TARGET quala-obj PROPERTY POSITION_INDEPENDENT_CODE 1)
target_include_directories(quala-obj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(quala-obj PUBLIC Eigen3::Eigen)

add_library(quala)
target_link_libraries(quala PUBLIC quala-obj)
add_library(quala::quala ALIAS quala)
set_target_properties(quala PROPERTIES EXPORT_NAME quala::quala)

# Target installation
# -------------------

if (NOT SKBUILD)
    include(cmake/Install.cmake)
endif()

# Python bindings
# ---------------

option(WITH_PYTHON_BINDINGS "Build the Python module" Off)

if (WITH_PYTHON_BINDINGS OR SKBUILD)
    add_subdirectory(quala)
endif()