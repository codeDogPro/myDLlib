cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)

project(tests LANGUAGES CXX)

add_executable(test src/test/test1.cc)

target_include_directories(test PUBLIC include)
