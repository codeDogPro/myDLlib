#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl {

  // TODO: have bugs
  template<typename T>
  __global__ void
  Conv2D_cuda(thrust::device_ptr<const T> input,
              thrust::device_ptr<T> output,
              thrust::device_ptr<T> weight,
              thrust::device_ptr<T> bias,
              const int k_size,
              const int stride,
              const int irow, 
              const int icol, 
              const int ich, 
              const int num,
              const int orow,
              const int ocol) {
    __shared__ T s_input[TILE_Y][TILE_X];
    __shared__ T s_weight[K_SIZE][K_SIZE];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x * TILE_X + tx;
    const int idx_y = blockIdx.y * TILE_Y + ty;
    
    if(idx_x < icol && idx_y < irow){
      //* The flag is used to choose which thread is needed
      const bool flag = (idx_x % stride == 0 && idx_y % stride == 0);

      //* conv all number and channel
      const T _bias = bias[blockIdx.z];
      for(int n = 0; n < num; n++){
        int ioffset = n * irow*icol*ich;
        int koffset = blockIdx.z * k_size*k_size*ich;
        T res = 0;
        for(int ch = 0; ch < ich; ch++){
          //* load input to shared memory
          const int iidx = idx_y*icol + idx_x;
          s_input[ty][tx] = input[ioffset + iidx];
          //* load weight to shared memory
          if(ty < k_size && tx < k_size){
            const int kidx = ty*k_size  + tx;
            s_weight[ty][tx] = weight[koffset + kidx];
          }
          __syncthreads();

          if(flag && idx_x + k_size <= icol && idx_y + k_size <= irow){
            for(int krow = 0; krow < k_size; krow++){   // *kernel row idx
              for(int kcol = 0; kcol < k_size; kcol++){ // *kernel col idx
                int in_y = ty + krow;
                int in_x = tx + kcol;
                if(in_x < TILE_X && in_y < TILE_Y){
                  res += s_weight[krow][kcol] * s_input[in_y][in_x]; 
                } else{ // *边界情况
                  res += s_weight[krow][kcol] * input[ioffset + in_y*icol + in_x];
                }
              } 
            }
          }
          ioffset += irow * icol;
          koffset += k_size * k_size;
        }
        //* add result and bias to output 
        // TODO: bug   The k_size > stride will make idx < 0
        const int oidx_x = idx_x / stride;
        const int oidx_y = idx_y / stride;
        const int ooffset = n*orow*ocol*gridDim.z + blockIdx.z*orow*ocol; 
        const int oidx = ooffset + oidx_y*ocol + oidx_x;
        if(flag && oidx_y < orow && oidx_x < ocol){
          // atomicAdd(output.get() + oidx, res + _bias);
          output[oidx] = res + _bias;
        }
      }
    }
  }

  template<typename T>
  __global__ void
  Conv2D_1x1_cuda(thrust::device_ptr<const T> input,
                  thrust::device_ptr<T> output,
                  thrust::device_ptr<T> weight,
                  thrust::device_ptr<T> bias,
                  const int stride,
                  const int irow, 
                  const int icol, 
                  const int ich, 
                  const int num,
                  const int orow,
                  const int ocol) {
    __shared__ T s_input[TILE_Y][TILE_X];
    __shared__ T s_weight;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x * TILE_X + tx;
    const int idx_y = blockIdx.y * TILE_Y + ty;
    
    if(idx_x < icol && idx_y < irow){
      //* The flag is used to choose which thread is needed
      const bool flag = (idx_x % stride == 0 && idx_y % stride == 0);

      //* conv all number and channel
      const T _bias = bias[blockIdx.z];
      for(int n = 0; n < num; n++){
        int ioffset = n * irow*icol*ich;
        int koffset = blockIdx.z * ich;
        T res = 0;
        for(int ch = 0; ch < ich; ch++){
          //* load input to shared memory
          const int iidx = idx_y*icol + idx_x;
          s_input[ty][tx] = input[ioffset + iidx];
          //* load weight to shared memory
          if(ty == 0 && tx == 0){
            s_weight = weight[koffset ++];
          }
          __syncthreads();

          if(flag && idx_x < icol && idx_y < irow){
            if(tx < TILE_X && ty < TILE_Y){
              res += s_weight * s_input[ty][tx]; 
            } else{ // *边界情况
              res += s_weight * input[ioffset + ty*icol + tx];
            }
          } 
          ioffset += irow * icol;
        }
        //* add result and bias to output 
        const int oidx_x = idx_x / stride;
        const int oidx_y = idx_y / stride;
        const int ooffset = n*orow*ocol*gridDim.z + blockIdx.z*orow*ocol; 
        const int oidx = ooffset + oidx_y*ocol + oidx_x;
        if(flag && oidx_y < orow && oidx_x < ocol){
          output[oidx] = res + _bias;
        }
      }
    }
  }
} // namespace dl