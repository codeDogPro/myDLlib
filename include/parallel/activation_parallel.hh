#pragma once

#include <basic/tensor_macro.hh>
#include <cmath>

namespace dl{

  template<typename T>
  bool relu_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    for(int i = start; i < end; i++){
      (*output)[i] = (*input)[i] > 0 ? (*input)[i] : 0;
    }
    return true;
  }

  template<typename T>
  bool sigmoid_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    for(int i = start; i < end; i++){
      (*output)[i] = 1 / (1 + std::exp(-(*input)[i])) + eps;
    }
    return true;
  }
}