#pragma once

#include <basic/tensor_macro.cuh>

namespace dl{

  template<typename T=f32>
  __global__ void 
  cuda_add_full(Tensor<T> &lhs, const Tensor<T> &rhs, T output){

  }

  template<typename T=f32>
  __global__ void 
  cuda_sub_full
  (const Tensor<T> &lhs, const Tensor<T> &rhs, std::shared_ptr<Tensor<T>> output){

  }

  template<typename T=f32>
  __global__ void 
  cuda_mul_full
  (const Tensor<T> &lhs, const Tensor<T> &rhs, std::shared_ptr<Tensor<T>> output){

  }

  template<typename T=f32>
  __global__ void 
  cuda_div_full
  (const Tensor<T> &lhs, const Tensor<T> &rhs, std::shared_ptr<Tensor<T>> output){

  }
}
