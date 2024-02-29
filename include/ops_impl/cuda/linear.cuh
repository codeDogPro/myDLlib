#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl{

  /*
  a: input
  b: weight
  */
  template<typename T=f32>
  __global__ void
  Linear_cuda(thrust::device_ptr<const T> a,
              thrust::device_ptr<const T> b,
              thrust::device_ptr<const T> bias,
              thrust::device_ptr<T> output, 
              const int acol,
              const int bcol){
    __shared__ T s_a[TILE_Y][TILE_X];
    __shared__ T s_b[TILE_Y][TILE_X];

    const int by = blockIdx.y, bx = blockIdx.x;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int row = by*TILE_Y + ty;
    const int col = bx*TILE_Y + tx;

    T res = 0;
    for(int ph = 0; ph < (acol + TILE_X-1)/TILE_X; ph++){
      // load global memory data to shared memory
      const int col_idx = tx + ph*TILE_X;
      const int row_idx = ty + ph*TILE_Y;
      int a_idx = blockIdx.z*acol + tx + ph*TILE_X;
      int b_idx = col + (ph*TILE_X + ty)*bcol;
      s_a[ty][tx] = (col_idx < acol && row < 1) ? a[a_idx] : static_cast<T>(0);
      s_b[ty][tx] = (col < bcol && row_idx < acol) ? b[b_idx] : static_cast<T>(0);
      __syncthreads();

      // matrix computation
      const int col_end = min(acol, TILE_X);
      for(int k = 0; k < col_end; k++){
        res += s_a[ty][k] * s_b[k][tx];
      }
      __syncthreads();
    }
    const int ooffset = blockIdx.z * bcol;
    if(col < bcol && row < 1){
      output[ooffset + col] = res + bias[col];
    }
  }

}