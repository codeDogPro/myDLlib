#pragma once

#include <basic/function.cuh>

#include <memory>
#include <parallel/activation_cpu.cuh>
#include <parallel/activation_cuda.cuh>

namespace dl{
  
template<typename T=f32>
class Relu : public Function<T> {
public:
  explicit Relu() = default;
  virtual ~Relu() = default;

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
    return (input->device() == Device::CPU) ? forward_cpu(input) : forward_cuda(input);
  }

private:
  std::shared_ptr<Tensor<T>> 
  forward_cpu(const std::shared_ptr<Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), 0);
    int row = input->row(), col = input->col(), channel = input->channel();
    int number = input->number(), volume = row * col * channel;
    if(channel > 1){
      // in conv layer
      for(int i = 0; i < number; i++){
        int offset = i * volume;
        parallelizer.parallel_channel(
          relu_parallel<T>, output, offset, input);
      }
    }
    else{
      // in linear layer
      parallelizer.parallel_col(
        relu_parallel<T>, output, 0, input);
    }
    parallelizer.sync();
    return output;
  }

  std::shared_ptr<Tensor<T>> 
  forward_cuda(const std::shared_ptr<Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(
      input->get_cshape(), 0, Device::CUDA);

    int gride_size = 64, block_size = 128;
    int size = output->size();
    printf("size: %d\n", size);
    relu_cuda<T><<<gride_size, block_size>>>
    (input->data_gpu(), output->data_gpu(), size);

    return output;
  }

};

}