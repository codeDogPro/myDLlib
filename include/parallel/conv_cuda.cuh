#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl {
  // TODO: have bugs
  template<typename T>
  __global__ void
  conv2D_cuda(thrust::device_ptr<T> input, thrust::device_ptr<T> output,
              thrust::device_ptr<T> weight, thrust::device_ptr<T> bias,
              int k_size, int stride,
              int irow, int icol, int ich, int inum){
    __shared__ T s_input[TILE_Y][TILE_X];
    __shared__ T s_weight[K_SIZE][K_SIZE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int orow = (irow - k_size)/stride + 1;
    int ocol = (icol - k_size)/stride + 1;
    int idx_x = blockIdx.x * TILE_X + tx;
    int idx_y = blockIdx.y * TILE_Y + ty;
    int ivolume = irow * icol * ich;
    int koffset = blockIdx.z * k_size*k_size*ich;

    // conv all number and channel
    for(int n = 0; n < inum; n++){
      int ioffset = n * ivolume;
      int ooffset = n * (orow*ocol*blockDim.z);
      int oidx = ooffset + idx_y*ocol + idx_x;
      for(int ch = 0; ch < ich; ch++){
        int iidx = ioffset + idx_y*icol + idx_x;
        int kidx = koffset + ty*k_size  + tx;
        s_weight[ty][tx] = (ty < k_size && tx < k_size) ? weight[kidx] : static_cast<T>(0);
        __syncthreads();
        s_input[ty][tx] = (idx_x < icol && idx_y < irow) ? input[iidx] : static_cast<T>(0);
        __syncthreads();

        if(idx_x < icol && idx_y < irow){
          T res = static_cast<T>(0);
          for(int krow = 0; krow < k_size; krow++){   // kernel row idx
            for(int kcol = 0; kcol < k_size; kcol++){ // kernel col idx
              int in_y = ty - k_size/2 + krow;
              int in_x = tx - k_size/2 + kcol;
              if(in_x >= 0 && in_x < TILE_X && in_y >= 0 && in_y < TILE_Y){
                res += s_weight[krow][kcol] * s_input[ty + krow][tx + kcol]; 
              } else{
                int edge_y = idx_y - k_size/2 + krow;
                int edge_x = idx_x - k_size/2 + kcol;
                if(edge_y >= 0 && edge_y < irow && edge_x >= 0 && edge_x < icol){
                  res += s_weight[krow][kcol] * input[ioffset + edge_y*icol + edge_x];
                }
              }
            } 
          }
          // add result to output
          atomicAdd(output.get() + oidx, res);
          // std::printf("res:%f\n", res);
        }
        ioffset += irow * icol;
        koffset += k_size * k_size;
      }
      // add bias
      // if(idx_x < ocol && idx_y < orow){
      //   atomicAdd(output.get() + oidx, bias[blockIdx.z]);
      // }
    }
  }

} // namespace dl