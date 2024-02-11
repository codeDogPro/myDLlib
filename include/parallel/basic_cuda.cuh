#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl{
  constexpr int _TILE_Y = 32;
  constexpr int _TILE_X = 32;
  template <typename T=f32>
  __global__ void 
  reduce4D_axis0_cuda(thrust::device_ptr<T> input, thrust::device_ptr<T> output, 
                int n, int col){
    __shared__ T sums[_TILE_Y][_TILE_X];
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int idx_x = bx * _TILE_X + tx;
    int idx = by * col * _TILE_Y + ty * col + idx_x;

    sums[ty][tx] = (idx_x < col && idx < n) ? input[idx] : static_cast<T>(0);
    __syncthreads();

    for(int offset = 16; offset > 0; offset >>= 1){
      if(tx < offset){
        sums[ty][tx] += sums[ty][tx + offset];
      }
      __syncwarp();  // could change to __syncwarp()?
    }

    int oidx = by * _TILE_Y + ty;
    if(tx == 0 && oidx < (n / col)){
      atomicAdd(output.get() + oidx, sums[ty][0]);
    }
  }
}