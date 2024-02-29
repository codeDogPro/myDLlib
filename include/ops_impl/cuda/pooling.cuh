#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl{

  template<typename T=f32>
  __global__ void
  __div_square(thrust::device_ptr<T> data, int square, int n){
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      data[i] = data[i] / square;
    }
  }

  template<typename T=f32>
  __global__ void
  globalAvgPool2D_cuda(thrust::device_ptr<const T> input,
                       thrust::device_ptr<T> output,
                       const int square){
    __shared__ T s_input[128];
    const int by = blockIdx.y, bx = blockIdx.x, tx = threadIdx.x;
    const int idx = bx * blockDim.x + tx;

    s_input[tx] = (idx < square) ? input[by*square + idx] : static_cast<T>(0); 
    __syncthreads();

    for(int ofst = blockDim.x >> 1; ofst >= 32; ofst >>= 1){
      if(tx < ofst){
        s_input[tx] += s_input[tx + ofst];
      }
      __syncthreads();
    }

    T value = s_input[tx];
    for(int ofst = 16; ofst > 0; ofst >>= 1){
      value += __shfl_down_sync(static_cast<ui32>(-1), value, ofst);
    }

    if(tx == 0){
      atomicAdd(output.get() + by, value); 
    }
  }


  template<typename T=f32>
  __global__ void
  AvgPool2D_cuda(thrust::device_ptr<const T> input,
                 thrust::device_ptr<T> output,
                 const int stride,
                 const int k_size,
                 const int irow, 
                 const int icol){
    __shared__ T s_input[TILE_Y][TILE_X];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x * TILE_X + tx;
    const int idx_y = blockIdx.y * TILE_Y + ty;
    
    const int x_stride = (idx_x == 0) ? 0 : stride - 1;
    const int y_stride = (idx_y == 0) ? 0 : stride - 1;
    if(idx_x < icol && idx_y < irow){
      const int orow = (irow - k_size)/stride + 1;
      const int ocol = (icol - k_size)/stride + 1;
      int ioffset = blockIdx.z * irow*icol;

      // load input to shared memory
      int iidx = idx_y*icol + idx_x;
      s_input[ty][tx] = input[ioffset + iidx];
      __syncthreads();

      T res = 0;
      const int edge = stride - 1 + k_size;
      if(idx_x + edge <= icol && idx_y + edge <= irow){
        for(int krow = 0; krow < k_size; krow++){   // kernel row idx
          for(int kcol = 0; kcol < k_size; kcol++){ // kernel col idx
            int in_y = ty + y_stride + krow;
            int in_x = tx + x_stride + kcol;
            if(in_x < TILE_X && in_y < TILE_Y){
              res += s_input[in_y][in_x];
            } else{ // 边界情况
              res += input[ioffset + in_y*icol + in_x];
            }
          } 
        }
      }
      // add result and bias to output 
      int oidx = blockIdx.z * orow*ocol + idx_y*ocol + idx_x;
      if(idx_y < orow && idx_x < ocol){
        atomicAdd(output.get() + oidx, res / (k_size*k_size));
      }
    }
  }

  template<typename T=f32>
  __global__ void
  MaxPool2D_cuda(thrust::device_ptr<const T> input,
                 thrust::device_ptr<T> output,
                 const int stride,
                 const int k_size,
                 const int irow, 
                 const int icol){
    __shared__ T s_input[TILE_Y][TILE_X];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int idx_x = blockIdx.x*TILE_X + tx;
    const int idx_y = blockIdx.y*TILE_Y + ty;
    
    const int x_stride = (idx_x == 0) ? 0 : stride - 1;
    const int y_stride = (idx_y == 0) ? 0 : stride - 1;
    if(idx_x < icol && idx_y < irow){
      const int ioffset = blockIdx.z * irow*icol;
      // load input to shared memory
      int iidx = idx_y*icol + idx_x;
      s_input[ty][tx] = input[ioffset + iidx];
      __syncthreads();

      T res = static_cast<T>(-1e8);
      const int edge = stride - 1 + k_size;
      if(idx_x + edge <= icol && idx_y + edge <= irow){
        for(int krow = 0; krow < k_size; krow++){   // kernel row idx
          for(int kcol = 0; kcol < k_size; kcol++){ // kernel col idx
            int in_y = ty + y_stride + krow;
            int in_x = tx + x_stride + kcol;
            if(in_x < TILE_X && in_y < TILE_Y){
              res = max(res, s_input[in_y][in_x]);
            } else{ // 边界情况
              res = max(res, input[ioffset + in_y*icol + in_x]);
            }
          } 
        }
      }
      // add result and bias to output 
      const int orow = (irow - k_size)/stride + 1;
      const int ocol = (icol - k_size)/stride + 1;
      int oidx = blockIdx.z * orow*ocol + idx_y*ocol + idx_x;
      if(idx_y < orow && idx_x < ocol){
        atomicAdd(output.get() + oidx, res);
      }
    }
  }
}