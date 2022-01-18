find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}"
        -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE quala_pybind11_include_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE QUALA_PYBIND11_RESULT)

if (QUALA_PYBIND11_RESULT EQUAL 0)
    message(STATUS "Found pybind11: ${quala_pybind11_include_dir}")
    add_library(pybind11::pybind11 INTERFACE IMPORTED)
    target_include_directories(pybind11::pybind11
        INTERFACE ${quala_pybind11_include_dir})

    Python3_add_library(_quala MODULE quala.cpp)
    target_link_libraries(_quala
        PRIVATE
            quala::quala-obj
            pybind11::pybind11
    )
    if (NOT WIN32)
        target_link_libraries(_quala 
            PRIVATE
                -static-libstdc++
                -static-libgcc
        )
    endif()
    target_compile_definitions(_quala PRIVATE VERSION_INFO=${PROJECT_VERSION})
    set(QUALA_RPATH "\${ORIGIN}/../../../.." "\${ORIGIN}")
    set_target_properties(_quala PROPERTIES
        DEBUG_POSTFIX ""
        ASAN_POSTFIX ""
        TSAN_POSTFIX ""
        CXX_VISIBILITY_PRESET hidden
        C_VISIBILITY_PRESET hidden
        INSTALL_RPATH "${QUALA_RPATH}"
    )
    if (NOT WIN32)
        target_link_options(_quala PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    target_compile_definitions(_quala PRIVATE QUALA_MODULE_NAME=_quala)

    option(QUALA_GEN_STUB "Generate Python stub files (.pyi) for the quala module." On)

    if (QUALA_GEN_STUB)
        find_program(STUBGEN_EXE stubgen REQUIRED PATH_SUFFIXES Scripts)
        set(STUBS_DIR ${CMAKE_CURRENT_BINARY_DIR}/stubs)
        add_custom_command(TARGET _quala POST_BUILD 
            COMMAND
                ${STUBGEN_EXE} 
                    -m quala
                    -o ${STUBS_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
            BYPRODUCTS ${STUBS_DIR}/quala/__init__.pyi
            USES_TERMINAL)
        add_custom_command(TARGET _quala POST_BUILD
            COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen
                    _quala
                    --ignore-invalid=defaultarg
                    --no-setup-py
                    -o ${STUBS_DIR}
            BYPRODUCTS ${STUBS_DIR}/_quala-stubs/__init__.pyi
            WORKING_DIRECTORY $<TARGET_FILE_DIR:_quala>
            USES_TERMINAL)
    endif()

    if (SKBUILD)
        install(TARGETS _quala
                DESTINATION .)
        if (QUALA_GEN_STUB)
            install(FILES ${STUBS_DIR}/_quala-stubs/__init__.pyi
                RENAME _quala.pyi
                DESTINATION .)
            install(FILES ${STUBS_DIR}/quala/__init__.pyi
                DESTINATION .)
        endif()
    endif()
else()
    message(WARNING "Could NOT find pybind11.")
endif()
