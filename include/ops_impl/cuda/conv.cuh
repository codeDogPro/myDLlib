#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl {
  // TODO: 
  template<typename T>
  __global__ void
  conv2D_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output,
              thrust::device_ptr<T> weight, thrust::device_ptr<T> bias,
              const int k_size, const int stride,
              const int irow, const int icol, const int ich, const int inum){
    __shared__ T s_input[TILE_Y][TILE_X];
    __shared__ T s_weight[K_SIZE][K_SIZE];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x * TILE_X + tx;
    const int idx_y = blockIdx.y * TILE_Y + ty;
    
    const int x_stride = (idx_x == 0) ? 0 : stride - 1;
    const int y_stride = (idx_y == 0) ? 0 : stride - 1;
    if(idx_x < icol && idx_y < irow){
      const int orow = (irow - k_size)/stride + 1;
      const int ocol = (icol - k_size)/stride + 1;
      // output channel offset
      int ooffset = blockIdx.z * orow*ocol; 

      int iidx = idx_y * icol + idx_x;
      int kidx = ty * k_size  + tx;
      // conv all number and channel
      for(int n = 0; n < inum; n++){
        int ioffset = n * irow*icol*ich;
        int koffset = blockIdx.z * k_size*k_size*ich;
        T res = 0;
        for(int ch = 0; ch < ich; ch++){
          // load input to shared memory
          s_input[ty][tx] = input[ioffset + iidx];
          // load weight to shared memory
          if(ty < k_size && tx < k_size){
            s_weight[ty][tx] = weight[koffset + kidx];
          }
          __syncthreads();

          const int edge = stride - 1 + k_size;
          if(idx_x + edge <= icol && idx_y + edge <= irow){
            for(int krow = 0; krow < k_size; krow++){   // kernel row idx
              for(int kcol = 0; kcol < k_size; kcol++){ // kernel col idx
                int in_y = ty + y_stride + krow;
                int in_x = tx + x_stride + kcol;
                if(in_x < TILE_X && in_y < TILE_Y){
                  res += s_weight[krow][kcol] * s_input[in_y][in_x]; 
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
  }

} // namespace dl