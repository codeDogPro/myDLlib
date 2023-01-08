#pragma once

#include <data/tensor.hh>

namespace dl{
  
  /*
    Every model function must inherit this class and override its apis
  */
  template<typename T>
  class Function{
  public:
    virtual Tensor<T> forward(const Tensor<T> &) = 0;
    // virtual Tensor<T> backward(); 
    virtual ~Function(){}
  };
 

}