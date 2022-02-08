if (WIN32)

    # Interpret source code as UTF-8, and define the "not" alternative operator
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /Dnot=!")

    # Build-type specific flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DDEBUG \
        /DEIGEN_INITIALIZE_MATRICES_BY_NAN")

else()

    # Build-type specific flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
        -DEIGEN_INITIALIZE_MATRICES_BY_NAN -DDEBUG")
    
endif()
set(CMAKE_DEBUG_POSTFIX "-debug")