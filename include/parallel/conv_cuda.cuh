#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl {
  // TODO: tomorrow try to implement conv_cuda
  template<typename T>
  __global__ void
  padding_cuda(thrust::device_ptr<T> input, thrust::device_ptr<T> output,
               int n, int icol, int irow, int npad){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int isquare = irow * icol;
    int offset = (2*(idx/isquare) + 1) 
                  * ((npad-1)*(icol+2*npad) + (icol+npad))  // 上下的横行
                 + (idx/isquare + idx/icol + 1)*2*npad;     // 左后的竖列
    if(idx < n){
      output[offset + idx] = input[idx];
    }
  }
}