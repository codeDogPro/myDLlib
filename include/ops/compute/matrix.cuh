#pragma once

#include "basic/tensor_macro.cuh"
#include <data/tensor.cuh>

#include <ops_impl/cpu/matrix.hh>
#include <ops_impl/cuda/matrix.cuh>

namespace dl{
  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matMul_cpu(const std::shared_ptr<const Tensor<T>> a,
             const std::shared_ptr<const Tensor<T>> b){
    const int arow = a->row(), bcol = b->col(), ch = a->channel(), num = a->number();

    auto output = std::make_shared<Tensor<T>>(arow, bcol, ch, num, Device::CPU, 0);
    for(int i = 0; i < ch * num; i++){
      parallelizer.parallel_row(matMul4D_cpu<T>, output,
        i, a, b);
    }
    parallelizer.sync();
    return output;
  }

  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matMul_cuda(const std::shared_ptr<const Tensor<T>> a, 
              const std::shared_ptr<const Tensor<T>> b){
    const int arow = a->row(), acol = a->col(), bcol = b->col();
    const int ch = a->channel(), num = a->number();

    auto output = std::make_shared<Tensor<T>>(arow, bcol, ch, num, Device::CUDA, 0);
    const int asize = a->size(), bsize = b->size(), osize = output->size();
    dim3 grid_size((arow + TILE_Y-1) / TILE_Y,
                   (acol + TILE_X-1) / TILE_X,
                   ch * num);
    dim3 block_size(TILE_Y, TILE_X);
    thrust::device_ptr<const T> _a = a->data_gpu(), _b = b->data_gpu();
    thrust::device_ptr<T> _output = output->data_gpu();
    matMul4D_cuda<<<grid_size, block_size>>>
      (_a, _b, _output, asize, bsize, osize, acol, arow, bcol);
    cudaDeviceSynchronize();
    return output;
  }

  /*
    This function will automatically check Tensor's data location,
    and invoke matched ops.
  */
  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matMul(const std::shared_ptr<const Tensor<T>> a,
         const std::shared_ptr<const Tensor<T>> b){
    const int acol = a->col(), achannel = a->channel();
    const int brow = b->row(), bchannel = b->channel();
    const int anumber = a->number(), bnumber = b->number();
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
  matTranspose_cpu(const std::shared_ptr<const Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), Device::CPU, 0);
    const int row = input->row(), col = input->col(), channel = input->channel();
    const int number = input->number(), volume = row * col * channel;
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
  matTranspose_cuda(const std::shared_ptr<const Tensor<T>> input){
    const int row = input->row(), col = input->col();
    const int ch = input->channel(), num = input->number();

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
  matTranspose(const std::shared_ptr<const Tensor<T>> input){
    return (input->device()==Device::CPU) ? matTranspose_cpu(input) : matTranspose_cuda(input);
  }
}