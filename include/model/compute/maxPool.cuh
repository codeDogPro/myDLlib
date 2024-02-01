#pragma once

#include <basic/function.cuh>
#include <basic/tensor_macro.cuh>

#include <parallel/conv_cpu.cuh>
#include <parallel/conv_cuda.cuh>

#include <parallel/pooling_cpu.cuh>
#include <parallel/pooling_cuda.cuh>

namespace dl{

// #define POOL_DEBUG

template<typename T=f32>
class MaxPool2D : public Function<T> {
public:
  explicit MaxPool2D() = default;
  virtual ~MaxPool2D() = default;

  explicit
  MaxPool2D(int pooling_size=2, int padding=0, int stride=-1){
    M_pool_size = pooling_size;
    if(stride == -1) M_stride = pooling_size;
    else             M_stride = stride;
    M_padding = padding;
  }

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> input){
    if(M_padding){
      auto pad_input = padding(input);
      return pool_boost(pad_input);
    }
    return pool_boost(input);
  }

private:
  std::shared_ptr<Tensor<T>>
  padding(const std::shared_ptr<Tensor<T>> input){
    int row = input->row(), col = input->col();
    int channel = input->channel(), number = input->number();
    auto pad_input = std::make_shared<Tensor<T>>
    (row + 2 * M_padding, col + 2 * M_padding, channel, number, 0);
    for(int i = 0; i < number; i++){
      int offset = i * number;
      parallelizer.parallel_channel(padding_parallel<T>,
        pad_input, offset, input, M_padding);
    }
    parallelizer.sync();
  #ifdef POOL_DEBUG
    std::cout << "pad_input:\n" << *pad_input << std::endl;
  #endif
    return pad_input;
  }

  int res_row(int row){return (row - M_pool_size) / M_stride + 1;}
  int res_col(int col){return (col - M_pool_size) / M_stride + 1;}

  std::shared_ptr<Tensor<T>>
  pool_boost(const std::shared_ptr<Tensor<T>> input){
    int irow = input->row(), icol = input->col();
    int channel = input->channel(), number = input->number();
    int orow = res_row(irow), ocol = res_col(icol);
    auto output = std::make_shared<Tensor<T>>(orow, ocol, channel, number, INT_MIN);
    for(int i = 0; i < number; i++){
      int offset = i * irow * icol * channel;
      parallelizer.parallel_channel(
        maxPooling_parallel<T>, output, offset, input, M_pool_size, M_stride);
    }
    parallelizer.sync();
    return output;
  }
  
private:
  int M_pool_size;
  int M_stride;
  int M_padding;
  };
}