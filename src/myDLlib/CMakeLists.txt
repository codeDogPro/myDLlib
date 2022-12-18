cmake_minimum_required(VERSION 3.22)

project(myDLlib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(myDLlib SHARED data/tensor.cpp)

target_include_directories(myDLlib PUBLIC ../../include)

find_package(Threads REQUIRED)
target_link_libraries(myDLlib PUBLIC Threads::Threads)

