cmake_minimum_required(VERSION 3.21)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

project(cuda_tests LANGUAGES CXX CUDA)

# 设置CUDA安装路径
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CMAKE_CUDA_COMPILER_ENV_VAR CUDA_PATH)
set(CMAKE_CUDA_COMPILER_VERSION 11.8)
set(CMAKE_CUDA_ARCHITECTURES 61)

# 开启avx2
add_definitions(-mavx2)


add_executable(cuda_test src/test/test2.cu)

# opencv lib
find_package(OpenCV REQUIRED)
target_include_directories(cuda_test PUBLIC ${OpenCV_INCLUDE_DIRS})

target_include_directories(cuda_test PUBLIC ./include)

target_link_libraries(cuda_test ${OpenCV_LIBS})

