#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl{

  template <typename T=f32>
  __global__ void 
  reduce4D_axis0_cuda(thrust::device_ptr<T> input, thrust::device_ptr<T> output, 
                int n, int col){
    __shared__ T sums[TILE_Y][TILE_X];
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int idx_x = bx * TILE_X + tx;
    int idx = by * col * TILE_Y + ty * col + idx_x;

    sums[ty][tx] = (idx_x < col && idx < n) ? input[idx] : static_cast<T>(0);
    __syncthreads();

    for(int offset = 16; offset > 0; offset >>= 1){
      if(tx < offset){
        sums[ty][tx] += sums[ty][tx + offset];
      }
      __syncwarp();  // could change to __syncwarp()?
    }

    int oidx = by * TILE_Y + ty;
    if(tx == 0 && oidx < (n / col)){
      atomicAdd(output.get() + oidx, sums[ty][0]);
    }
  }

  /*tensor initialization kernel for defferent data type*/
  __global__ void 
  tensor_init_i32(thrust::device_ptr<i32> data){

  }

  __global__ void 
  tensor_init_f32(thrust::device_ptr<f32> data){

  }
}