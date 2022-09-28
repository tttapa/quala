cmake_minimum_required(VERSION 3.21)
project(quala-test CXX)
enable_testing()
include(GoogleTest)

# Dependencies
find_package(GTest REQUIRED CONFIG)
if (NOT TARGET quala::quala)
    find_package(quala REQUIRED CONFIG)
endif()

# Test executable compilation and linking
add_executable(tests
    "eigen-matchers.hpp"
    "test-alloc.cpp"
    "test-anderson-acceleration.cpp"
    "test-lbfgs.cpp"
    "test-limited-memory-qr.cpp"
    "test-ringbuffer.cpp"
)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(tests PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(tests PRIVATE quala::quala 
                                    GTest::gtest_main 
                                    GTest::gmock)

gtest_discover_tests(tests DISCOVERY_TIMEOUT 60)
add_executable(quala::tests ALIAS tests)
                                    
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.21) 
    # Copy DLLs for test executable
    add_custom_command(TARGET tests PRE_LINK COMMAND
        COMMAND ${CMAKE_COMMAND} -E 
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:tests>>,copy,true>
            $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
        COMMAND_EXPAND_LISTS)
endif()
