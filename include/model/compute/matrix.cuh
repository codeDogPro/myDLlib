#pragma once

#include "basic/tensor_macro.cuh"
#include <cuda_device_runtime_api.h>
#include <data/tensor.cuh>

#include <parallel/matrix_cpu.cuh>
#include <parallel/matrix_cuda.cuh>

namespace dl{
  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matMul_cpu(const std::shared_ptr<Tensor<T>> a, const std::shared_ptr<Tensor<T>> b){
    int arow = a->row(), bcol = b->col(), ch = a->channel(), num = a->number();

    auto output = std::make_shared<Tensor<T>>(arow, bcol, ch, num, Device::CPU, 0);
    if(num == 1 && ch == 1){
      parallelizer.parallel_row(matMul_row_cpu<T>, output,
        0, a, b);
    }
    else{
      for(int i = 0; i < num; i++){
        parallelizer.parallel_channel(matMul_channel_cpu<T>, output,
          i, a, b);
      }
    }
    parallelizer.sync();
    return output;
  }

  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matMul_cuda(const std::shared_ptr<Tensor<T>> a, const std::shared_ptr<Tensor<T>> b){
    int arow = a->row(), acol = a->col(), bcol = b->col();
    int ch = a->channel(), num = a->number();

    auto output = std::make_shared<Tensor<T>>(arow, bcol, ch, num, Device::CUDA, 0);
    auto _a = a->data_gpu(), _b = b->data_gpu(), _output = output->data_gpu();
    dim3 grid_size(arow / TILE_Y, acol / TILE_X, ch * num);
    dim3 block_size(TILE_Y, TILE_X);
    matMul_cuda<<<grid_size, block_size>>>
      (_a, _b, _output, acol, arow, bcol);
    cudaDeviceSynchronize();
    return output;
  }

  /*
    This function will automatically check Tensor's data location,
    and invoke matched ops.
  */
  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matMul(const std::shared_ptr<Tensor<T>> a, const std::shared_ptr<Tensor<T>> b){
    int acol = a->col(), achannel = a->channel();
    int brow = b->row(), bchannel = b->channel();
    int anumber = a->number(), bnumber = b->number();
    assert(achannel == bchannel && anumber == bnumber);

    if(a->device() == b->device()){
      if(acol == brow){
        return (a->device() == Device::CPU) ? matMul_cpu(a, b) : matMul_cuda(a, b);
      }
      else {
        fprintf(stderr, "mat a's col != mat b's row!\n");
        exit(-1);
      }
    }
    else{
      fprintf(stderr, "Tensor must be in the same device!\n");
      exit(-1);
    }
  }

  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matTranspose_cpu(const std::shared_ptr<Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), Device::CPU, 0);
    int row = input->row(), col = input->col(), channel = input->channel();
    int number = input->number(), volume = row * col * channel;
    for(int i = 0; i < number; i++){
      int offset = i * volume;
      parallelizer.parallel_channel(matTranspose_cpu<T>, output,
        offset, input);
    }
    parallelizer.sync();
    return output;
  }

  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matTranspose_cuda(const std::shared_ptr<Tensor<T>> input){
    int row = input->row(), col = input->col();
    int ch = input->channel(), num = input->number();

    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), Device::CUDA, 0);
    auto _output = output->data_gpu(), _input = input->data_gpu();
    dim3 grid_size(col / TILE_X, row / TILE_Y, ch * num);
    dim3 block_size(TILE_Y, TILE_X);
    matTranspose4D_cuda<<<grid_size, block_size>>>
      (_input, _output, row, col);
    return output;
  }

  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matTranspose(const std::shared_ptr<Tensor<T>> input){
    return (input->device()==Device::CPU) ? matTranspose_cpu(input) : matTranspose_cuda(input);
  }
}