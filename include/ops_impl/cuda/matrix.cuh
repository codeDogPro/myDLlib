#pragma once

#include <basic/tensor_macro.cuh>

#include <thrust/device_ptr.h>

namespace dl{

  template<typename T=f32>
  __global__ void
  matMul4D_cuda(thrust::device_ptr<const T> a,
                thrust::device_ptr<const T> b,
                thrust::device_ptr<T> output, 
                const int asize,
                const int bsize,
                const int osize,
                const int acol,
                const int arow,
                const int bcol){
    __shared__ T s_a[TILE_Y][TILE_X];
    __shared__ T s_b[TILE_Y][TILE_X];

    const int by = blockIdx.y, bx = blockIdx.x;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int row = by*TILE_Y + ty;
    const int col = bx*TILE_Y + tx;

    const int aoffset = blockIdx.z * arow*acol;
    const int boffset = blockIdx.z * acol*bcol;

    T res = 0;
    for(int ph = 0; ph < (acol + TILE_X-1)/TILE_X; ph++){
      // load global memory data to shared memory
      const int col_idx = tx + ph*TILE_X;
      const int row_idx = ty + ph*TILE_Y;
      int a_idx = aoffset + row*acol + tx + ph*TILE_X;
      int b_idx = boffset + col + (ph*TILE_X + ty)*bcol;
      s_a[ty][tx] = (col_idx < acol && row < arow) ? a[a_idx] : static_cast<T>(0);
      s_b[ty][tx] = (col < bcol && row_idx < acol) ? b[b_idx] : static_cast<T>(0);
      __syncthreads();

      // matrix computation
      const int col_end = min(acol, TILE_X);
      for(int k = 0; k < col_end; k++){
        res += s_a[ty][k] * s_b[k][tx];
      }
      __syncthreads();
    }
    const int ooffset = blockIdx.z * arow*bcol;
    if(col < bcol && row < arow){
      output[ooffset + row*bcol + col] = res;
    }
  }

  template<typename T=f32>
  __global__ void
  matTranspose4D_cuda(thrust::device_ptr<const T> input,
                      thrust::device_ptr<T> output,
                      const int row,
                      const int col){
    __shared__ T s_m[TILE_Y][TILE_X + 1];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int offset = blockIdx.z * row*col;

    const int nx1 = blockIdx.x*TILE_X + tx;
    const int ny1 = blockIdx.y*TILE_Y + ty;
    if(nx1 < col && ny1 < row){
      s_m[ty][tx] = input[offset + ny1*col + nx1];
    }
    __syncthreads();

    const int nx2 = blockIdx.x*TILE_X + ty;
    const int ny2 = blockIdx.y*TILE_Y + tx;
    if(nx2 < col && ny2 < row){
      output[offset + nx2*col + ny2] = s_m[tx][ty];
    }
  }
}