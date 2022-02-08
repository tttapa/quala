cmake_minimum_required(VERSION 3.17)
project(quala-dev LANGUAGES CXX)

# Import file with some custom compiler warnings
include(cmake/GlobalCompilerWarnings.cmake)

# ----

# Source code
add_subdirectory(src)

# Unit tests
option(QUALA_WITH_TESTS
    "Compile the unit tests" On) 
if (QUALA_WITH_TESTS)
    find_package(GTest REQUIRED CONFIG)
    include(GoogleTest)
    enable_testing()
    add_subdirectory(test)
endif()

# Examples
option(QUALA_WITH_EXAMPLES
    "Compile the examples" On)
if (QUALA_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()

# ----

# Add documentation targets
add_custom_target(documentation-doxygen
    doxygen
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/doxygen)
add_custom_target(documentation-doxygen-breathe
    doxygen Doxyfile.breathe
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/doxygen)
add_custom_target(documentation-sphinx-doctest
    sphinx-build -b doctest
        ${CMAKE_CURRENT_LIST_DIR}/sphinx/source 
        ${CMAKE_CURRENT_LIST_DIR}/docs/Sphinx
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
add_dependencies(documentation-sphinx-doctest documentation-doxygen-breathe)
add_custom_target(documentation-sphinx
    sphinx-build -M html
        ${CMAKE_CURRENT_LIST_DIR}/sphinx/source 
        ${CMAKE_CURRENT_LIST_DIR}/docs/Sphinx
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
add_dependencies(documentation-sphinx documentation-doxygen-breathe)
add_custom_target(documentation)
add_dependencies(documentation documentation-doxygen documentation-sphinx)

# Add coverage target
option(QUALA_WITH_COVERAGE
    "Generate coverage information." Off)
if (QUALA_WITH_COVERAGE)
    add_custom_target(coverage
        ${CMAKE_CURRENT_LIST_DIR}/scripts/coverage.sh
        ${CMAKE_CXX_COMPILER_ID}
        ${CMAKE_CXX_COMPILER_VERSION}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fno-inline")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-inline")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE On)
    add_dependencies(coverage quala::tests)
endif()

# ----

# Packaging
include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${quala_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${quala_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${quala_VERSION_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_GENERATOR "DEB;TGZ")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Pieter P")
include(CPack)
