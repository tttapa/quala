# Make sure that the Python and CMake versions match
set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})
if (DEFINED VERIFY_VERSION)
    if (NOT "${VERIFY_VERSION}" MATCHES "^${PY_FULL_VERSION}$")
        message(FATAL_ERROR "Version number does not match "
                             "(${VERIFY_VERSION} - ${PY_FULL_VERSION}).")
    endif()
endif()

# Find the Pybind11 headers
include(cmake/QueryPythonForPybind11.cmake)
find_pybind11_python_first()

# Create Python module
Python3_add_library(_quala MODULE quala.py.cpp)
target_link_libraries(_quala
    PRIVATE
        quala::quala-obj
        pybind11::pybind11)
set_target_properties(_quala PROPERTIES
    DEBUG_POSTFIX ""
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden)
target_compile_definitions(_quala 
    PRIVATE
        QUALA_MODULE_NAME=_quala
        QUALA_VERSION_INFO="${PY_FULL_VERSION}")
if (NOT WIN32)
    target_link_options(_quala PRIVATE "LINKER:--exclude-libs,ALL")
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