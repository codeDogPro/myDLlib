#pragma once

#include <data/tensor.hpp>

namespace dl{
  
  /*
    Every model function must inherit this class and override its apis
  */
  template<typename T>
  class Function{
  public:
    virtual Tensor<T> forward(Tensor<T> &) = 0;
    // virtual Tensor<T> backward(); 
  };
 

}