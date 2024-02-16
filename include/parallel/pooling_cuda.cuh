#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl{

  template<typename T=f32>
  __global__ void
  globalAvgPool2D_cuda
  (thrust::device_ptr<T> input, thrust::device_ptr<T> output, int n, int square){
    __shared__ T s_input[128];
    const int by = blockIdx.y, bx = blockIdx.x, tx = threadIdx.x;
    const int idx = bx * blockDim.x + tx;

    s_input[tx] = (idx < square) ? input[by*square + idx] : static_cast<T>(0); 
    __syncthreads();

    for(int ofst = blockDim.x >> 1; ofst >= 32; ofst >>= 1){
      if(tx < ofst){
        s_input[tx] += s_input[tx + ofst];
      }
    }

    T value = s_input[tx];
    for(int ofst = 16; ofst > 0; ofst >>= 1){
      value += __shfl_down_sync(static_cast<ui32>(-1), value, ofst);
    }

    if(tx == 0 && idx < square){
      // atomicAdd(output.get() + by, value / square); 
      atomicAdd(output.get() + by, value); 
    }
  }

}