#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <data/tensor.h>

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
#endif