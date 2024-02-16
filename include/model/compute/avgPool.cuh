#pragma once

#include <basic/function.cuh>
#include <basic/tensor_macro.cuh>

#include <cuda_device_runtime_api.h>
#include <parallel/conv_cpu.cuh>
#include <parallel/conv_cuda.cuh>

#include <parallel/pooling_cpu.cuh>
#include <parallel/pooling_cuda.cuh>

namespace dl{
 

  template<typename T=f32>
  class AvgPool2D : public Function<T> {
public:
  explicit AvgPool2D() = default;
  virtual ~AvgPool2D() = default;

  explicit
  AvgPool2D(int pooling_size=2, int padding=0, int stride=-1){
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
    auto output = std::make_shared<Tensor<T>>(orow, ocol, ch, num, Device::CUDA, 0);
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
    (row + 2 * M_padding, col + 2 * M_padding, ch, num, Device::CPU, 0);
    for(int i = 0; i < num; i++){
      int offset = i * num;
      parallelizer.parallel_channel(padding_cpu<T>,
        pad_input, offset, input, M_padding);
    }
    parallelizer.sync();
    return pad_input;
  }

  int res_row(int row){return (row - M_pool_size) / M_stride + 1;}
  int res_col(int col){return (col - M_pool_size) / M_stride + 1;}

  std::shared_ptr<Tensor<T>>
  pool_cpu(const std::shared_ptr<Tensor<T>> input){
    int irow = input->row(), icol = input->col();
    int ch = input->channel(), num = input->number();
    int orow = res_row(irow), ocol = res_col(icol);
    auto output = std::make_shared<Tensor<T>>(orow, ocol, ch, num, Device::CPU, 0);
    for(int i = 0; i < num; i++){
      int offset = i * irow * icol * ch;
      parallelizer.parallel_channel(
        avgPool2D_cpu<T>, output, offset,
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



  template<typename T=f32>
  class globalAvgPool2D : public Function<T> {
public:
  explicit globalAvgPool2D() = default;
  virtual ~globalAvgPool2D() = default;

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> input){
    return (input->device() == Device::CPU) ? forward_cpu(input) : forward_cuda(input);
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<Tensor<T>> input){
    return forward(input);
  }


private:
  std::shared_ptr<Tensor<T>>
  forward_cuda(const std::shared_ptr<Tensor<T>> input){
    int row = input->row(), col = input->col();
    int size = input->size(), square = row * col;
    int ch = input->channel(), num = input->number();
    auto output = std::make_shared<Tensor<T>>(1, 1, ch, num, Device::CUDA, 0);

    auto _input = input->data_gpu(), _output = output->data_gpu();
    dim3 grid_size((square + 127)/128, num * ch), block_size(128);
    // printf("grid_x: %d grid_y:%d\n", grid_size.x, grid_size.y);
    globalAvgPool2D_cuda<<<grid_size, block_size>>> (_input, _output, square);
    __div_square<<<16, 128>>>(_output, square, ch * num);
    cudaDeviceSynchronize();
    return output; 
  }

  std::shared_ptr<Tensor<T>>
  forward_cpu(const std::shared_ptr<Tensor<T>> input){
    int ch = input->channel(), num = input->number();
    // TODO: implement it
    auto output = std::make_shared<Tensor<T>>(1, 1, ch, num, Device::CPU, 0);
    return output;
  }

  };
}