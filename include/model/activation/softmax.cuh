#pragma once

#include <basic/function.cuh>

#include <parallel/activation_cpu.cuh>
#include <parallel/activation_cuda.cuh>

namespace dl{

template<typename T=f32>
class Softmax : public Function<T> {
public:
  explicit Softmax(int axis=0) {
    M_axis = Axis(axis);
  };
  virtual ~Softmax() = default;

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
    if(input->device() == Device::CPU){
      return forward_cpu(input);
    }
    else{
      return forward_cuda(input);
    }
  }

private:
  std::shared_ptr<Tensor<T>> 
  forward_cpu(const std::shared_ptr<Tensor<T>> input){
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

    // then get softmax
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

  std::shared_ptr<Tensor<T>> 
  forward_cuda(const std::shared_ptr<Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), 0);
    int row = input->row(), col = input->col(), channel = input->channel();
    int size = output->size();

    auto _input = input->data_gpu(), _output = output->data_gpu();
    expf_cuda(_input, _output, size);
    if(M_axis == Axis::COL){
      int gride_size = 64, block_size = 128;
      softmax_axis0_cuda(_input, _output, size, row, col);
    }
    else if(M_axis == Axis::ROW){
      int gride_size = 64, block_size = 128;
      softmax_axis1_cuda(_input, _output, size);
    }
    else if(M_axis == Axis::CHANNEL){
      int gride_size = 64, block_size = 128;
      softmax_axis2_cuda(_input, _output, size);
    }
    return output;
  }

  Axis M_axis;
};
}