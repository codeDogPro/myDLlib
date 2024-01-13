cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

project(tests LANGUAGES CXX)

# 开启avx2
add_definitions(-mavx2)

add_executable(test src/test/test1.cc)

target_include_directories(test PUBLIC include)
