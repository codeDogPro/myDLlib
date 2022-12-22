cmake_minimum_required(VERSION 3.22)

project(myDLlib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

file(GLOB source_files data/*.cpp)

add_library(myDLlib SHARED ${source_files})

target_include_directories(myDLlib PUBLIC ../../include)

find_package(Threads REQUIRED)
target_link_libraries(myDLlib PUBLIC Threads::Threads)

