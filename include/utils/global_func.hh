#pragma once

#include <data/tensor.hh>

namespace dl{

  /*
  Flatten make tensor shape: 1x1x2048 --> 1x2048x1,
  which to match the Linear layer.
  */
  template<typename T>
  inline void 
  flatten(const Tensor<T>& tensor){
    int channel = tensor.channel(), number = tensor.number();
    tensor.reshape(1, channel, 1, number);
  }
}