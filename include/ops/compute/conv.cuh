#pragma once

#include "basic/tensor_macro.cuh"
#include <basic/function.cuh>

#include <ops_impl/cpu/conv.hh>
#include <ops_impl/cpu/basic.hh>
#include <ops_impl/cuda/conv.cuh>
#include <ops_impl/cuda/basic.cuh>


namespace dl{

// #define CONV_DEBUG_WEIGHT
// #define CONV_DEBUG_PAD
template<typename T=f32>
class Conv2D : public Function<T> {
public:
  explicit 
  Conv2D (int kernel_size, int input_ch, int output_ch, 
          int stride=1, int padding=0, Device device=Device::CPU)
  : M_weight(kernel_size, kernel_size, input_ch, output_ch, device),
    M_bias(output_ch, 1, 1, 1, device)
  {
    M_stride     = stride;
    M_padding    = padding;
    M_kernelSize = kernel_size;
    m_device     = device;
  #ifdef CONV_DEBUG_WEIGHT
    std::cout << "weight:\n" << M_weight << std::endl;
    std::cout << "bias:\n" << M_bias << std::endl;
  #endif
  }

  explicit 
  Conv2D(Tensor<T>& weight, Tensor<T> &bias, int stride=1, int padding=0) {
    m_device = weight.device();
    M_weight = std::move(weight);
    M_bias   = std::move(bias);
    M_stride     = stride;
    M_padding    = padding;
    M_kernelSize = M_weight.row();
  #ifdef CONV_DEBUG_WEIGHT
    std::cout << "weight:\n" << M_weight << std::endl;
    std::cout << "bias:\n" << M_bias << std::endl;
  #endif
  }

  virtual ~Conv2D() = default;

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<const Tensor<T>> input){
    return (input->device() == Device::CPU) ? forward_cpu(input) : forward_cuda(input);
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<const Tensor<T>> input){
    return forward(input);
  }

  void to(Device device){
    M_weight.to(device);
    M_bias.to(device);
    m_device = device;
  }

  int nstride()  { return M_stride; }
  int npadding() { return M_padding; }

private:
  int res_row(int row){return (row - M_kernelSize + 2*M_padding)/M_stride + 1;}
  int res_col(int col){return (col - M_kernelSize + 2*M_padding)/M_stride + 1;}

  std::shared_ptr<Tensor<T>> 
  forward_cuda(const std::shared_ptr<const Tensor<T>> input){
    if(M_padding){ // need to pad
      auto pad_input = _pad_cuda(input);
      return _conv_cuda(pad_input);
    } else{         // no padding
      return _conv_cuda(input);
    }
  }

  std::shared_ptr<const Tensor<T>> 
  _pad_cuda(const std::shared_ptr<const Tensor<T>> input){
    const int row = input->row(), col = input->col();
    const int ch = input->channel(), num = input->number();
    const int pad_row = row + 2*M_padding, pad_col = col + 2*M_padding;
    auto output = std::make_shared<Tensor<T>>
      (pad_row, pad_col, ch, num, Device::CUDA, 0);
    thrust::device_ptr<const T> _input = input->data_gpu();
    thrust::device_ptr<T> _output = output->data_gpu();
    const int size = input->size();
    const int grid_size = (size + 127) / 128, block_size = 128;
    padding_cuda<<<grid_size, block_size>>>
      (_input, _output, size, row, col, M_padding);
    cudaDeviceSynchronize();
  #ifdef CONV_DEBUG_PAD
    std::cout << "pad_input:\n" << *output;
  #endif
    return output;
  }

  std::shared_ptr<Tensor<T>> 
  _conv_cuda(const std::shared_ptr<const Tensor<T>> input){
    const int ich = input->channel(), och = M_weight.number();
    const int irow = input->row(), icol = input->col(), num = input->number();
    const int orow = (irow-M_kernelSize)/M_stride + 1;
    const int ocol = (icol-M_kernelSize)/M_stride + 1;
    auto output = std::make_shared<Tensor<T>>(orow, ocol, och, num, Device::CUDA, 0);

    thrust::device_ptr<const T> _input = input->data_gpu();
    thrust::device_ptr<T> _output = output->data_gpu();
    auto _weight = M_weight.data_gpu(), _bias = M_bias.data_gpu();

    const int grid_x = (icol+TILE_X-1)/TILE_X, grid_y = (irow+TILE_Y-1)/TILE_Y;
    dim3 grid_size(grid_x, grid_y, och);
    dim3 block_size(TILE_X, TILE_Y);

    if(M_kernelSize == 1){
      Conv2D_1x1_cuda<<<grid_size, block_size>>>
        (_input, _output, _weight, _bias,
         M_stride, irow, icol, ich, num, orow, ocol); 
    } else{
      Conv2D_cuda<<<grid_size, block_size>>>
        (_input, _output, _weight, _bias,
        M_kernelSize, M_stride,
        irow, icol, ich, num, orow, ocol); 
    }
    cudaDeviceSynchronize();
    return output;
  }

  std::shared_ptr<Tensor<T>> 
  forward_cpu(const std::shared_ptr<const Tensor<T>> input){
    const int row = input->row(), col = input->col();
    const int ch = input->channel(), num = input->number();
    if(M_padding){
      auto pad_input = std::make_shared<Tensor<T>>
      (row + 2*M_padding, col + 2*M_padding, ch, num, Device::CPU, 0);
      const int ivolume = row * col * ch;
      for(int i = 0; i < num; i++){
        const int offset = i * ivolume;
        parallelizer.parallel_channel(padding_cpu<T>,
         pad_input, offset, input, M_padding);
      }
      parallelizer.sync();
  #ifdef CONV_DEBUG_PAD
      std::cout << "pad_input:\n" << *pad_input << std::endl;
  #endif
      return _conv_cpu(pad_input, res_row(row), res_col(col));
    } else{
      return _conv_cpu(input, res_row(row), res_col(col));
    }
  }

  std::shared_ptr<Tensor<T>> 
  _conv_cpu(const std::shared_ptr<const Tensor<T>> input, int orow, int ocol){
    const int irow = input->row(), icol = input->col(), ich = input->channel();
    const int num = input->number(), ivolume = irow * icol * ich;
    const int och = M_weight.number();
    auto output = std::make_shared<Tensor<T>>(orow, ocol, och, num, Device::CPU, 0);
    for(int i = 0; i < num; i++){
      const int offset = i * ivolume;
      if(M_kernelSize == 1){
        parallelizer.parallel_channel(conv2d_1x1_cpu<T>, output, offset, input, 
          M_weight, M_bias, M_stride);
      } else{
        parallelizer.parallel_channel(conv2d_cpu<T>, output, offset, input, 
          M_weight, M_bias, M_stride);
      }
    }
    parallelizer.sync();
    return output;
  }

  int M_padding;
  int M_stride;
  int M_kernelSize;
  Device m_device;
  Tensor<T> M_weight, M_bias;
};

}