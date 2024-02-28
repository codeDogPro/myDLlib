#pragma once

#include <basic/tensor_macro.cuh>

#include <thrust/device_ptr.h>

namespace dl{

  template<typename T=f32>
  __global__ void
  matMul4D_cuda(thrust::device_ptr<const T> a, thrust::device_ptr<const T> b,
                thrust::device_ptr<T> output, 
                int acol, int arow, int bcol){
    __shared__ T s_a[TILE_Y][TILE_X];
    __shared__ T s_b[TILE_Y][TILE_X];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_Y + ty;
    int col = bx * TILE_Y + tx;

    int aoffset = blockIdx.z * arow * acol;
    int boffset = blockIdx.z * acol * bcol;
    T res = static_cast<T>(0);
    for(int ph = 0; ph < acol / TILE_X; ph++){
      // load global memory data to shared memory
      s_a[ty][tx] = a[aoffset + row * acol + ph *TILE_X + tx];
      s_b[ty][tx] = b[boffset + (ph * TILE_X + ty) * acol + col];
      __syncthreads();

      // matrix computation
      for(int k = 0; k < TILE_X; k++){
        res += s_a[ty][k] * s_b[k][tx];
      }
      __syncthreads();
    }
    int ooffset = blockIdx.z * arow * bcol;
    output[ooffset + row * acol + col] = res;
  }

  template<typename T=f32>
  __global__ void
  matTranspose4D_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output,
                      int row, int col){
    __shared__ T s_m[TILE_Y][TILE_X + 1];
    int ty = threadIdx.y, tx = threadIdx.x;
    int offset = blockIdx.z * row * col;

    int nx1 = blockIdx.x * TILE_X + tx;
    int ny1 = blockIdx.y * TILE_Y + ty;
    if(nx1 < col && ny1 < row){
      s_m[ty][tx] = input[offset + ny1 * col + nx1];
    }
    __syncthreads();

    int nx2 = blockIdx.x * TILE_X + ty;
    int ny2 = blockIdx.y * TILE_Y + tx;
    if(nx2 < col && ny2 < row){
      output[offset + nx2 * col + ny2] = s_m[tx][ty];
    }
  }
}