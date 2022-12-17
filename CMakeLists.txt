cmake_minimum_required(VERSION 3.22)

project(tests LANGUAGES CXX)

add_subdirectory(src/myDLlib)

add_executable(test src/test/test1.cpp)

target_link_libraries(test PUBLIC myDLlib)

target_include_directories(test PUBLIC include)