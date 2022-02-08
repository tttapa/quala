find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Found Python ${Python3_VERSION}")

# Test executable compilation and linking
add_executable(tests
    "eigen-matchers.hpp"
    "test-alloc.cpp"
    "test-anderson-acceleration.cpp"
    "test-lbfgs.cpp"
    "test-limited-memory-qr.cpp"
    "test-ringbuffer.cpp"
)
target_include_directories(tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(tests
    PRIVATE
        GTest::gtest_main
        GTest::gmock
        quala::quala
)
add_executable(quala::tests ALIAS tests)

# Add tests
if (NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(tests)
endif()