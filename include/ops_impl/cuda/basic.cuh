#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl{

template <typename T = f32>
__global__ void 
reduce4D_axis0_cuda(thrust::device_ptr<T> input,
                    thrust::device_ptr<T> output,
                    const int n,
                    const int col) {
  __shared__ T sums[TILE_Y][TILE_X];
  const int by = blockIdx.y, bx = blockIdx.x;
  const int ty = threadIdx.y, tx = threadIdx.x;
  const int idx_x = bx * TILE_X + tx;
  uint64_t idx = by * col * TILE_Y + ty * col + idx_x;

  sums[ty][tx] = (idx_x < col && idx < n) ? input[idx] : static_cast<T>(0);
  __syncthreads();

  for (int offset = 16; offset > 0; offset >>= 1) {
    if (tx < offset) {
      sums[ty][tx] += sums[ty][tx + offset];
    }
    __syncwarp(); // could change to __syncwarp()?
  }

  const uint64_t oidx = by * TILE_Y + ty;
  if (tx == 0 && oidx < (n / col)) {
    atomicAdd(output.get() + oidx, sums[ty][0]);
  }
}

  template<typename T>
  __global__ void
  padding_cuda(thrust::device_ptr<const T> input,
               thrust::device_ptr<T> output,
               const int n,
               const int icol,
               const int irow,
               const int npad){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const int isquare = irow * icol;
    /*上下的横行*/                        /* npad-1个完整行         一个非完整行*/
    const uint64_t offset = (2*(idx/isquare) + 1) * ((npad-1)*(icol+2*npad) + (icol+npad))
    /*左右的竖列*/ + (idx/isquare + idx/icol + 1)*2*npad;     
    if(idx < n){
      output[offset + idx] = input[idx];
    }
  }
}