#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl {
  // TODO: fix bugs
  template<typename T>
  __global__ void
  conv2D_cuda(thrust::device_ptr<T> input, thrust::device_ptr<T> output,
              thrust::device_ptr<T> weight, thrust::device_ptr<T> bias,
              int k_size, int stride,
              int irow, int icol, int ich, int inum){
    __shared__ volatile T s_input[TILE_Y][TILE_X];
    __shared__ volatile T s_weight[K_SIZE][K_SIZE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int orow = (irow - k_size)/stride + 1;
    int ocol = (icol - k_size)/stride + 1;
    int idx_x = blockIdx.x * TILE_X + tx;
    int idx_y = blockIdx.y * TILE_Y + ty;
    int iidx = idx_y * icol + idx_x;
    int kidx = ty * k_size  + tx;
    // output channel offset
    int ooffset = blockIdx.z * orow*ocol; 
    
    // conv all number and channel
    for(int n = 0; n < inum; n++){
      int ioffset = n * irow*icol*ich;
      int koffset = blockIdx.z * k_size*k_size*ich;
      T res = 0;
      for(int ch = 0; ch < ich; ch++){
        // load weight to shared memory
        if(ty < k_size && tx < k_size){
          s_weight[ty][tx] = weight[koffset + kidx];
          // std::printf("weight[%d][%d]: %.1f idx_x:%d idx_y:%d ch:%d\n",
          //   ty, tx, s_weight[ty][tx], idx_x, idx_y, ch);
        }
        // load input to shared memory
        if(idx_x < icol && idx_y < irow){
          s_input[ty][tx] = input[ioffset + iidx];
          // std::printf("input[%d][%d]: %.1f idx_x:%d, idx_y:%d ch:%d\n",
          //  ty, tx, s_input[ty][tx], idx_x, idx_y, ch);
        }
        __syncthreads();

        if(idx_x + k_size <= icol && idx_y + k_size <= irow){
          for(int krow = 0; krow < k_size; krow++){   // kernel row idx
            for(int kcol = 0; kcol < k_size; kcol++){ // kernel col idx
              int in_y = ty + krow;
              int in_x = tx + kcol;
              if(in_x < TILE_X && in_y < TILE_Y){
                res += s_weight[krow][kcol] * s_input[in_y][in_x]; 
                // std::printf("res:%.1f weight[%d][%d]: %.1f input[%d][%d]: %.1f idx_x:%d idx_y:%d ch:%d\n",
                //  res, krow, kcol, s_weight[krow][kcol], in_y, in_x, s_input[in_y][in_x], idx_x, idx_y, ch);
              } else{ // 边界情况
                res += s_weight[krow][kcol] * input[ioffset + in_y*icol + in_x];
              }
            } 
          }
        }
        ioffset += irow * icol;
        koffset += k_size * k_size;
      }
      // add result and bias to output 
      int oidx = ooffset + idx_y*ocol + idx_x;
      if(idx_y < orow && idx_x < ocol){
        atomicAdd(output.get() + oidx, res + bias[blockIdx.z]);
      }

      // add number offset to output offset
      ooffset += orow * ocol * gridDim.z;
    }
  }

} // namespace dl