#pragma once

#include <basic/function.cuh>
#include <basic/tensor_macro.cuh>

#include <ops_impl/cpu/conv.hh>
#include <ops_impl/cpu/basic.hh>
#include <ops_impl/cpu/pooling.hh>
#include <ops_impl/cuda/pooling.cuh>

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
  forward(const std::shared_ptr<const Tensor<T>> input){
    return (input->device() == Device::CPU) ? forward_cpu(input) : forward_cuda(input);
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<const Tensor<T>> input){
    return forward(input);
  }

private:
  std::shared_ptr<Tensor<T>>
  forward_cuda(const std::shared_ptr<const Tensor<T>> input){
    if(M_padding){  // need to pad
      auto pad_input = _pad_cuda(input);
      return _pool_cuda(pad_input);
    } else{         // no padding
      return _pool_cuda(input);
    }
  }

  std::shared_ptr<const Tensor<T>> 
  _pad_cuda(const std::shared_ptr<const Tensor<T>> input){
    int row = input->row(), col = input->col();
    int ch = input->channel(), num = input->number();
    int pad_row = row + 2*M_padding, pad_col = col + 2*M_padding;
    auto output = std::make_shared<Tensor<T>>
      (pad_row, pad_col, ch, num, Device::CUDA, 0);
    thrust::device_ptr<const T> _input = input->data_gpu();
    thrust::device_ptr<T> _output = output->data_gpu();
    const int size = input->size();
    const int grid_size = (size + 127) / 128, block_size = 128;
    padding_cuda<<<grid_size, block_size>>>
      (_input, _output, size, row, col, M_padding);
  #ifdef CONV_DEBUG_PAD
    std::cout << "pad_input:\n" << *output;
  #endif
    return output;
  }

  std::shared_ptr<Tensor<T>>
  _pool_cuda(const std::shared_ptr<const Tensor<T>> input){
    const int ch = input->channel(), num = input->number();
    const int irow = input->row(), icol = input->col();
    const int orow = res_row(irow), ocol = res_col(icol);
    auto output = std::make_shared<Tensor<T>>(orow, ocol, ch, num, Device::CUDA, 0);

    thrust::device_ptr<const T> _input = input->data_gpu();
    thrust::device_ptr<T> _output = output->data_gpu();

    const int grid_x = (icol+TILE_X-1)/TILE_X, grid_y = (irow+TILE_Y-1)/TILE_Y;
    dim3 grid_size(grid_x, grid_y, ch * num);
    dim3 block_size(TILE_X, TILE_Y);
    MaxPool2D_cuda<<<grid_size, block_size>>>
      (_input, _output,
       M_pool_size, M_stride, irow, icol, orow, ocol); 
    return output; 
  }

  std::shared_ptr<Tensor<T>>
  forward_cpu(const std::shared_ptr<const Tensor<T>> input){
    if(M_padding){
      auto pad_input = _pad_cpu(input);
      return _pool_cpu(pad_input);
    }
    return _pool_cpu(input);
  }

  std::shared_ptr<Tensor<T>>
  _pad_cpu(const std::shared_ptr<const Tensor<T>> input){
    const int row = input->row(), col = input->col();
    const int ch = input->channel(), num = input->number();
    auto pad_input = std::make_shared<Tensor<T>>
    (row + 2 * M_padding, col + 2 * M_padding, ch, num, Device::CPU, 0);
    for(int i = 0; i < num; i++){
      const int offset = i * row*col*ch;
      parallelizer.parallel_channel(padding_cpu<T>,
        pad_input, offset, input, M_padding);
    }
    parallelizer.sync();
    return pad_input;
  }

  std::shared_ptr<Tensor<T>>
  _pool_cpu(const std::shared_ptr<const Tensor<T>> input){
    const int irow = input->row(), icol = input->col();
    const int ch = input->channel(), num = input->number();
    const int orow = res_row(irow), ocol = res_col(icol);
    auto output = std::make_shared<Tensor<T>>(orow, ocol, ch, num, Device::CPU, 0);
    for(int i = 0; i < num; i++){
      const int offset = i * irow * icol * ch;
      parallelizer.parallel_channel(
        maxPool2D_cpu<T>, output, offset,
        input, M_pool_size, M_stride);
    }
    parallelizer.sync();
    return output;
  }

  int res_row(int row){return (row - M_pool_size) / M_stride + 1;}
  int res_col(int col){return (col - M_pool_size) / M_stride + 1;}

  
  int M_pool_size;
  int M_stride;
  int M_padding;
  };
}