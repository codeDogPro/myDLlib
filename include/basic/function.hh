#pragma once

#include <data/tensor.hh>

namespace dl{
  
  /*
    Every model function must inherit this class and override its apis
  */
  template<typename T=f32>
  class Function{
  public:
    virtual std::shared_ptr<Tensor<T>> 
    forward(const std::shared_ptr<Tensor<T>>) = 0;
    // virtual Tensor<T> backward(); 
    virtual ~Function(){}
  };
 

}