#pragma once

#include <basic/tensor_macro.cuh>
#include <thrust/device_ptr.h>

namespace dl{

  template<typename T=f32>
  __global__ void
  globalAvgPool2D_cuda(thrust::device_ptr<const T> input, thrust::device_ptr<T> output,
                       int n, int square){
    // TODO: implement it

  }

}