#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>
#include <cstdio>

namespace dl{

  template <typename T=f32>
  __global__ void 
  relu_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output, int n) {
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      output[i] = (input[i] > static_cast<T>(0)) ? input[i] : static_cast<T>(0);
    }
  }

  template <typename T=f32>
  __global__ void 
  sigmoid_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output, int n) {
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      output[i] = 1.0 / (1.0 + expf(-input[i])) + eps;
    }
  }

  template <typename T=f32>
  __global__ void 
  exp_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output, int n){
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      output[i] = expf(input[i]);
    }
  }

  constexpr int _TILE_Y = 32;
  constexpr int _TILE_X = 32;
  template <typename T=f32>
  __global__ void 
  softmax_axis0_cuda(thrust::device_ptr<T> exp, thrust::device_ptr<T> exp_sum, 
                     thrust::device_ptr<T> output, int n, int col) {
    // TODO: fix bug! result is not right
    // calculate every row's sum
    __shared__ T sums[_TILE_Y][_TILE_X];
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int idx_x = bx * _TILE_X + tx;
    int idx = by * col * _TILE_Y + ty * col + idx_x;

    sums[ty][tx] = (idx_x < col && idx < n) ? exp[idx] : static_cast<T>(0);
    __syncthreads();

    for(int offset = 16; offset > 0; offset >>= 1){
      if(tx < offset){
        sums[ty][tx] += sums[ty][tx + 16];
      }
      __syncthreads();  // could change to __syncwarp()?
    }

    int oidx = by * _TILE_Y + ty;
    T *addr = exp_sum.get() + oidx;
    if(tx == 0 && oidx < n / col){
      atomicAdd(addr, sums[ty][0]);
    }
    // reduce finished

    //calculate the softmax
    __syncthreads();
    if(idx < n){
      output[idx] = exp[idx] / exp_sum[oidx];
    }
    // softmax finished
  }

  template <typename T=f32>
  __global__ void 
  softmax_axis1_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output,
                     int n, int row, int col, int channel) {
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // TODO: use shared memory
    for(int i = begin; i < n; i += stride){
    }
  }

  template <typename T=f32>
  __global__ void 
  softmax_axis2_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output,
                     int n, int row, int col, int channel) {
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // TODO: use shared memory
    for(int i = begin; i < n; i += stride){
    }
  }
}
