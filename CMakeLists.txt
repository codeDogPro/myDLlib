cmake_minimum_required(VERSION 3.21)

# Set CUDA path
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CMAKE_CUDA_COMPILER_ENV_VAR CUDA_PATH)
set(CMAKE_CUDA_COMPILER_VERSION 11.8)
set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_CXX_COMPILER "clang++")

project(cuda_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)


find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(cuda_test src/test/test2.cu)


# 设置cuda编译参数
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets )
  message(STATUS "${PROJECT_NAME}: compile cuda code in release mode")
else()
  set(NVCC_FLAGS -G  -Wno-deprecated-gpu-targets )
  message(STATUS "${PROJECT_NAME}: compile cuda code in debug mode")
endif()
# cuda compile flag
target_compile_options(cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${NVCC_FLAGS}>)

# 开启avx2
add_definitions(-mavx2)

target_include_directories(cuda_test PUBLIC ${OpenCV_INCLUDE_DIRS} ./include)

# link cuda toolkit lib and opencv lib
target_link_libraries(cuda_test ${OpenCV_LIBS} CUDA::curand)
