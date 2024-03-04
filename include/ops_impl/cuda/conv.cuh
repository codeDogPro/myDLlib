#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/detail/cstdint.h>
#include <thrust/device_ptr.h>

namespace dl {

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
    __shared__ T s_bias;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x * TILE_X + tx;
    const int idx_y = blockIdx.y * TILE_Y + ty;

    //* load bias to shared memory
    if(ty == 0 && tx == 0){
      s_bias = bias[blockIdx.z];
    }
    
    if(idx_x < icol && idx_y < irow){
      //* The flag is used to choose which thread is needed
      const bool flag = (idx_x % stride == 0 && idx_y % stride == 0);

      //* conv all number and channel
      for(int n = 0; n < num; n++){
        uint64_t ioffset = n * irow*icol*ich;
        uint64_t koffset = blockIdx.z * k_size*k_size*ich;
        T res = 0;
        for(int ch = 0; ch < ich; ch++){
          //* load input to shared memory
          const uint64_t iidx = idx_y*icol + idx_x;
          s_input[ty][tx] = input[ioffset + iidx];
          //* load weight to shared memory
          if(ty < k_size && tx < k_size){
            const uint64_t kidx = ty*k_size  + tx;
            s_weight[ty][tx] = weight[koffset + kidx];
          }
          __syncthreads();

          if(flag && idx_x + k_size <= icol && idx_y + k_size <= irow){
            for(int krow = 0; krow < k_size; krow++){   // *kernel row idx
              for(int kcol = 0; kcol < k_size; kcol++){ // *kernel col idx
                if(tx + kcol < TILE_X && ty + krow < TILE_Y){
                  res += s_weight[krow][kcol] * s_input[ty + krow][tx + kcol]; 
                } else{ // *边界情况
                  res += s_weight[krow][kcol] 
                          * input[ioffset + (idx_y+krow)*icol + idx_x + kcol];
                }
              } 
            }
          }
          ioffset += irow * icol;
          koffset += k_size * k_size;
        }
        //* add result and bias to output 
        const int oidx_x = idx_x / stride;
        const int oidx_y = idx_y / stride;
        const uint64_t ooffset = n*orow*ocol*gridDim.z + blockIdx.z*orow*ocol; 
        const uint64_t oidx = ooffset + oidx_y*ocol + oidx_x;
        if(flag && oidx_y < orow && oidx_x < ocol){
          // atomicAdd(output.get() + oidx, res + _bias);
          output[oidx] = res + s_bias;
        }
      }
    }
  }

  template<typename T>
  __global__ void
  Conv2D_k1_cuda(thrust::device_ptr<const T> input,
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
    __shared__ T s_bias;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x * TILE_X + tx;
    const int idx_y = blockIdx.y * TILE_Y + ty;
    
    //* load bias to shared memory
    if(ty == 0 && tx == 0){
      s_bias = bias[blockIdx.z];
    }
    
    if(idx_x < icol && idx_y < irow){
      //* The flag is used to choose which thread is needed
      const bool flag = (idx_x % stride == 0 && idx_y % stride == 0);

      //* conv all number and channel
      for(int n = 0; n < num; n++){
        uint64_t ioffset = n * irow*icol*ich;
        uint64_t koffset = blockIdx.z * ich;
        T res = 0;
        for(int ch = 0; ch < ich; ch++){
          //* load input to shared memory
          const uint64_t iidx = idx_y*icol + idx_x;
          s_input[ty][tx] = input[ioffset + iidx];
          //* load weight to shared memory
          if(ty == 0 && tx == 0){
            s_weight = weight[koffset ++];
          }
          __syncthreads();

          if(flag && idx_x < icol && idx_y < irow){
            res += s_weight * s_input[ty][tx]; 
          } 
          ioffset += irow * icol;
        }
        //* add result and bias to output 
        const int oidx_x = idx_x / stride;
        const int oidx_y = idx_y / stride;
        const uint64_t ooffset = n*orow*ocol*gridDim.z + blockIdx.z*orow*ocol; 
        const uint64_t oidx = ooffset + oidx_y*ocol + oidx_x;
        if(flag && oidx_y < orow && oidx_x < ocol){
          output[oidx] = res + s_bias;
        }
      }
    }
  }

  //* kernel size=1 and stride=1
  template<typename T>
  __global__ void
  Conv2D_k1s1_cuda(thrust::device_ptr<const T> input,
                  thrust::device_ptr<T> output,
                  thrust::device_ptr<T> weight,
                  thrust::device_ptr<T> bias,
                  const int row, 
                  const int col, 
                  const int ich, 
                  const int num) {
    __shared__ T s_input[TILE_Y][TILE_X];
    __shared__ T s_weight;
    __shared__ T s_bias;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x * TILE_X + tx;
    const int idx_y = blockIdx.y * TILE_Y + ty;
    const int square = row*col;

    //* load bias to shared memory
    if(ty == 0 && tx == 0){
      s_bias = bias[blockIdx.z];
    }
    
    if(idx_x < col && idx_y < row){
      //* conv all number and channel
      for(int n = 0; n < num; n++){
        uint64_t ioffset = n * square*ich;
        uint64_t koffset = blockIdx.z * ich;
        T res = 0;
        for(int ch = 0; ch < ich; ch++){
          //* load input to shared memory
          const uint64_t iidx = idx_y*col + idx_x;
          s_input[ty][tx] = input[ioffset + iidx];
          //* load weight to shared memory
          if(ty == 0 && tx == 0){
            s_weight = weight[koffset ++];
          }
          __syncthreads();
          res += s_weight * s_input[ty][tx]; 
          ioffset += square;
        }
        //* add result and bias to output 
        const uint64_t ooffset = n*square*gridDim.z + blockIdx.z*square; 
        const uint64_t oidx = ooffset + idx_y*col + idx_x;
        output[oidx] = res + s_bias;
      }
    }
  }
} // namespace dl