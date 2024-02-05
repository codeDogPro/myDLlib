#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

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
  softmax_axis0_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output,
                     int n, int row, int col) {
    int bid = blockIdx.x, tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;
    // sums's size is row * col;
    // TODO: use shared memory
    extern __shared__ T sums[];
    sums[tid] = (idx < n) ? input[idx] : static_cast<T>(0);
    __syncthreads();

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
}
