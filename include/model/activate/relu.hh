#pragma once

#include <basic/function.hh>
#include <data/tensor.hh>

namespace dl{
  template<typename T>
  class Relu : public Function<T> {
    explicit
    Relu();

    virtual Tensor<T> 
    forward(const Tensor<T> &input){
      
    }
  };
}