set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

# Make sure that the Python and CMake versions match
if (DEFINED PY_BUILD_CMAKE_PACKAGE_VERSION)
    if (NOT "${PY_BUILD_CMAKE_PACKAGE_VERSION}" MATCHES "^${PY_FULL_VERSION}$")
        message(FATAL_ERROR "Version number does not match "
                             "(${PY_BUILD_CMAKE_PACKAGE_VERSION} - ${PY_FULL_VERSION}).")
    endif()
endif()

# Find the Pybind11 headers
include(cmake/QueryPythonForPybind11.cmake)
find_pybind11_python_first()

# Create Python module
pybind11_add_module(_quala MODULE quala.py.cpp)
target_link_libraries(_quala
    PRIVATE
        quala::quala-obj
        pybind11::pybind11
        quala::warnings)
set_target_properties(_quala PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden)
target_compile_definitions(_quala 
    PRIVATE
        QUALA_MODULE_NAME=$<TARGET_FILE_BASE_NAME:_quala>
        QUALA_VERSION_INFO="${PY_FULL_VERSION}")
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(_quala PRIVATE "LINKER:--exclude-libs,ALL")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(_quala PRIVATE "/bigobj")
endif()

# Install the Python module
install(TARGETS _quala
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION quala)

# Generate stubs for the Python module
option(QUALA_WITH_PY_STUBS
    "Generate Python stub files (.pyi) for the Python module." On)
if (QUALA_WITH_PY_STUBS)
    include(cmake/Pybind11Stubgen.cmake)
    pybind11_stubgen(_quala)
    pybind11_stubgen_install(_quala quala)
endif()