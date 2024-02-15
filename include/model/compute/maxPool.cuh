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
    if(input->device() == Device::CPU){
      return forward_cpu(input);
    }
    else{
      return forward_cuda(input);
    }
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<Tensor<T>> input){
    return forward(input);
  }

private:
  std::shared_ptr<Tensor<T>>
  forward_cuda(const std::shared_ptr<Tensor<T>> input){
    int irow = input->row(), icol = input->col();
    int ch = input->channel(), num = input->number();
    int orow = res_row(irow), ocol = res_col(icol);
    auto output = std::make_shared<Tensor<T>>(orow, ocol, ch, num, 0, Device::CUDA);
    // xxxxx cuda kernel
    return output; 
  }

  std::shared_ptr<Tensor<T>>
  forward_cpu(const std::shared_ptr<Tensor<T>> input){
    if(M_padding){
      auto pad_input = pad_cpu(input);
      return pool_cpu(pad_input);
    }
    return pool_cpu(input);
  }

  std::shared_ptr<Tensor<T>>
  pad_cpu(const std::shared_ptr<Tensor<T>> input){
    int row = input->row(), col = input->col();
    int ch = input->channel(), num = input->number();
    auto pad_input = std::make_shared<Tensor<T>>
    (row + 2 * M_padding, col + 2 * M_padding, ch, num, 0);
    for(int i = 0; i < num; i++){
      int offset = i * num;
      parallelizer.parallel_channel(padding_cpu<T>,
        pad_input, offset, input, M_padding);
    }
    parallelizer.sync();
    return pad_input;
  }

  std::shared_ptr<Tensor<T>>
  pool_cpu(const std::shared_ptr<Tensor<T>> input){
    int irow = input->row(), icol = input->col();
    int ch = input->channel(), num = input->number();
    int orow = res_row(irow), ocol = res_col(icol);
    auto output = std::make_shared<Tensor<T>>(orow, ocol, ch, num, 0);
    for(int i = 0; i < num; i++){
      int offset = i * irow * icol * ch;
      parallelizer.parallel_channel(
        maxPool2D_cpu<T>, output, offset,
        input, M_pool_size, M_stride);
    }
    parallelizer.sync();
    return output;
  }

  int res_row(int row){return (row - M_pool_size) / M_stride + 1;}
  int res_col(int col){return (col - M_pool_size) / M_stride + 1;}

  
private:
  int M_pool_size;
  int M_stride;
  int M_padding;
  };
}