function(pybind11_stubgen target)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen
                ${target}
                --no-setup-py
                -o ${CMAKE_CURRENT_BINARY_DIR}
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${target}-stubs/__init__.pyi
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
        USES_TERMINAL)

endfunction()

function(pybind11_stubgen_install target destination)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${target}-stubs/__init__.pyi
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        RENAME ${target}.pyi
        DESTINATION ${destination})

endfunction()