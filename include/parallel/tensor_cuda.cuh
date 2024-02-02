#pragma once

#include <basic/tensor_macro.cuh>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace dl{

  template<typename T=f32>
  __global__ void 
  cuda_add_full(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      output[i] = lhs[i] + rhs[i];
    }
  }

  template<typename T=f32>
  __global__ void 
  cuda_sub_full(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      output[i] = lhs[i] - rhs[i];
    }
  }

  template<typename T=f32>
  __global__ void 
  cuda_mul_full(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      output[i] = lhs[i] * rhs[i];
    }
  }

  template<typename T=f32>
  __global__ void 
  cuda_div_full(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      output[i] = lhs[i] / rhs[i];
    }
  }

  template<typename T=f32>
  __global__ void 
  cuda_add_single(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){

  }

  template<typename T=f32>
  __global__ void 
  cuda_sub_single(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){

  }

  template<typename T=f32>
  __global__ void 
  cuda_mul_single(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){

  }

  template<typename T=f32>
  __global__ void 
  cuda_div_single(thrust::device_ptr<const T> lhs, thrust::device_ptr<const T> rhs, 
   thrust::device_ptr<T> output, int n){

  }
}
