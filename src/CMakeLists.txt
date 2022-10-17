# Eigen linear algebra library
find_package(Eigen3 REQUIRED)

# Quala library
# -------------

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
target_compile_features(quala-obj PUBLIC cxx_std_17)
add_library(quala::quala-obj ALIAS quala-obj)
target_include_directories(quala-obj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(quala-obj
    PUBLIC Eigen3::Eigen
    PRIVATE quala::warnings)

add_library(quala)
target_link_libraries(quala PUBLIC quala-obj)
add_library(quala::quala ALIAS quala)
set_target_properties(quala PROPERTIES EXPORT_NAME quala::quala)

# Target installation
# -------------------

include(${PROJECT_SOURCE_DIR}/cmake/Install.cmake)

# Python bindings
# ---------------

if (QUALA_WITH_PYTHON)
    add_subdirectory(python)
endif()