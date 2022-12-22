cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)

project(tests LANGUAGES CXX)

add_subdirectory(src/myDLlib)

add_executable(test src/test/test1.cpp)

target_include_directories(test PUBLIC include)

target_link_libraries(test PUBLIC myDLlib)
