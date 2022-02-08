if (WIN32)

    # Enable compiler warnings globally, interpret source code as UTF-8, and 
    # define the "not" alternative operator
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")

else()

    # Enable compiler warnings globally
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -Wall -Wextra -Werror \
        -pedantic -pedantic-errors \
        -fdiagnostics-show-option \
        -Wmissing-include-dirs \
        -Wno-unknown-pragmas \
        -Wno-error=unused-parameter \
        -Wno-error=unused-variable")
    
endif()
