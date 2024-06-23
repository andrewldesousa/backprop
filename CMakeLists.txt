cmake_minimum_required(VERSION 3.12)
project(backprop)

# debug build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(tests)


add_executable(and_example examples/and.cpp)