cmake_minimum_required(VERSION 3.22)

project(myDLlib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(myDLlib SHARED tensor.cpp)

target_include_directories(myDLlib PUBLIC ../../include)
