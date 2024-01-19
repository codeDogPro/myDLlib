#pragma once

#include <basic/function.hh>
#include <data/tensor.hh>
#include <parallel/activation_parallel.hh>

namespace dl{

template<typename T=f32>
class Softmax : public Function<T> {
public:
  explicit
  Softmax(int axis=0) {
    M_axis = Axis(axis);
  };

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), 0);
    int row = input->row(), col = input->col(), channel = input->channel();
    int number = input->number(), volume = row * col * channel;

    // get exp(input) first
    if(channel > 1){
      for(int i = 0; i < number; i++){
        int offset = i * volume;
        parallelizer.parallel_channel(
          exp_parallel<T>, input, offset, input);
      }
    }
    else if(channel == 1 && row == 1){
      // in linear output
      for(int i = 0; i < number; i++){
        int offset = i * volume;
        parallelizer.parallel_col(
          exp_parallel<T>, input, offset, input);
      }
    }
    parallelizer.sync();
    std::cout << "input:\n" << *input << std::endl;
    for(int i = 0; i < number; i++){
      int offset = i * volume;
      if(M_axis == Axis::COL){
        parallelizer.parallel_channel(
          softmax_axis0_parallel<T>, output, offset, input);
      }
      else if(M_axis == Axis::ROW){
        parallelizer.parallel_channel(
          softmax_axis1_parallel<T>, output, offset, input);
      }
      else if(M_axis == Axis::CHANNEL){
        parallelizer.parallel_row(
          softmax_axis2_parallel<T>, output, offset, input);
      }
    }
    parallelizer.sync();
    return output;
  }

private:
  Axis M_axis;
};
}