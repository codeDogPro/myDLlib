cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

project(tests LANGUAGES CXX)

# 开启avx2
add_definitions(-mavx2)

add_executable(test src/test/test1.cc)

find_package(OpenCV REQUIRED)
target_include_directories(test PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(test PUBLIC include)

target_link_libraries(test ${OpenCV_LIBS})