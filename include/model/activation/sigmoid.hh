#pragma once

#include <basic/function.hh>
#include <data/tensor.hh>
#include <parallel/activation_parallel.hh>

namespace dl{
  
template<typename T=f32>
class Sigmoid : public Function<T> {
public:
  explicit
  Sigmoid() = default;

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), 0);
    int row = input->row(), col = input->col(), channel = input->channel();
    int number = input->number(), volume = row * col * channel;
    if(channel > 1){
      // in conv layer
      for(int i = 0; i < number; i++){
        int offset = i * volume;
        parallelizer.parallel_channel(
          sigmoid_parallel<T>, output, offset, input);
      }
    }
    else{
      // in linear layer
      parallelizer.parallel_col(
          sigmoid_parallel<T>, output, 0, input);
    }
    parallelizer.sync();
    return output;
  }
};

}