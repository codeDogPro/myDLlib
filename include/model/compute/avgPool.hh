#pragma once

#include <basic/function.hh>
#include <basic/tensor_macro.hh>
#include <data/tensor.hh>
#include <parallel/conv_parallel.hh>
#include <parallel/pooling_parallel.hh>

namespace dl{

template<typename T=f32>
class AvgPool2D : public Function<T> {
public:
  explicit
  AvgPool2D() = default;

  explicit
  AvgPool2D(int pooling_size=2, int stride=1, int padding=0){
    M_pool_size = pooling_size;
    M_stride = stride;
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
    // std::cout << "pad_input:\n" << *pad_input << std::endl;
    return pad_input;
  }

  int res_row(int row){return (row - M_pool_size) / M_stride + 1;}
  int res_col(int col){return (col - M_pool_size) / M_stride + 1;}

  std::shared_ptr<Tensor<T>>
  pool_boost(const std::shared_ptr<Tensor<T>> input){
    int irow = input->row(), icol = input->col();
    int channel = input->channel(), number = input->number();
    int orow = res_row(irow), ocol = res_col(icol);
    auto output = std::make_shared<Tensor<T>>(orow, ocol, channel, number, 0);
    for(int i = 0; i < number; i++){
      int offset = i * irow * icol * channel;
      parallelizer.parallel_channel(
        avgPooling_parallel<T>, output, offset,
        input, M_pool_size, M_stride);
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