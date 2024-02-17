#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl {

  // TODO: try to implement conv_cuda
  template<typename T>
  __global__ void
  conv2D_cuda(thrust::device_ptr<T> input, thrust::device_ptr<T> output,
              thrust::device_ptr<T> weight, thrust::device_ptr<T> bias,
              int irow, int icol, int kernel_size, int stride){
    __shared__ T s_input[TILE_Y][TILE_X];
    extern __shared__ T s_weight[];
    extern __shared__ T s_bias[];
    int by = blockIdx.y, bx = blockIdx.x, bz = blockIdx.z;
    int ty = threadIdx.y, tx = threadIdx.x;
    int idx_x = bx * TILE_X + tx, idx_y = by * TILE_Y + ty;

    int ioffset = bz * irow * icol;
    int iidx = ioffset + idx_y * icol + idx_x;
    s_input[ty][tx] = (idx_x < icol && idx_y < irow) ? input[iidx] : static_cast<T>(0);
    __syncthreads();

    if(idx_x < icol && idx_y < irow){
      int orow = (irow - kernel_size) / stride + 1;
      int ocol = (icol - kernel_size) / stride + 1;
      T res = static_cast<T>(0);
      for(int krow = 0; krow < kernel_size; krow++){   // kernel row idx
        for(int kcol = 0; kcol < kernel_size; kcol++){ // kernel col idx
          int in_y = ty - kernel_size/2 + krow;
          int in_x = tx - kernel_size/2 + kcol;
          int kidx = krow * kernel_size + kcol;
          if(in_x >= 0 && in_x < TILE_X && in_y >= 0 && in_y < TILE_Y){
            res += s_weight[kidx] * s_input[ty + krow][tx + kcol]; 
          }
          else{
            int edge_y = idx_y - kernel_size/2 + krow;
            int edge_x = idx_x - kernel_size/2 + kcol;
            if(edge_y >= 0 && edge_y < irow && edge_x >= 0 && edge_x < icol){
              res += s_weight[kidx] * input[ioffset + edge_y*icol + edge_x];
            }
          }
        } 
        int ooffset = bz * orow * ocol;
        int oidx = ooffset + idx_y * ocol + idx_x;
        atomicAdd(output[oidx], res);
      }
    }
  }
}