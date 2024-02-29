#pragma once

#include <basic/function.cuh>
#include <basic/tensor_macro.cuh>

#include <parallel/parallel.cuh>
#include <ops_impl/cpu/linear.hh>
#include <ops_impl/cuda/linear.cuh>

namespace dl{

template<typename T=f32>
class Linear : public Function<T> {
public:
  explicit 
  Linear(int input_dim, int output_dim, Device device=Device::CPU) 
  : M_weight(output_dim, input_dim, 1, 1, device),
    M_bias(1, output_dim, 1, 1, device) {
    /*
      *For cuda impl, it's matMul, so we need to reshape M_weight.
      *[output_dim, input_dim] -> [input_dim, output_dim]
    */
    if(device == Device::CUDA){
      M_weight.reshape(input_dim, output_dim, 1, 1); 
    }
    // #define LINEAR_DEBUG
    #ifdef LINEAR_DEBUG
      std::cout << "weight:\n"<<M_weight << std::endl;
      std::cout << "bias:\n" << M_bias << std::endl;
      M_weight.to(Device::CUDA);
      M_bias.to(Device::CUDA);
    #endif
    } 
  
  virtual ~Linear() = default;

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<const Tensor<T>> input) override{
    return (input->device() == Device::CPU) ? forward_cpu(input) : forward_cuda(input);
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<const Tensor<T>> input){
    return forward(input);
  }

  void to(Device device){
    // M_weight need to reshape
    M_weight.reshape(M_weight.col(), M_weight.row(), 1, 1); 
    M_weight.to(device);
    M_bias.to(device);
    m_device = device;
  }

private:
  std::shared_ptr<Tensor<T>>
  forward_cuda(const std::shared_ptr<const Tensor<T>> input) {
    int icol = input->col(), ocol = M_bias.col(), num = input->number();
    auto output = std::make_shared<Tensor<T>>(1, ocol, 1, num, Device::CUDA, 0);
    dim3 grid_size((icol + TILE_X-1) / TILE_X, 1, num);
    dim3 block_size(TILE_X, TILE_Y);
    thrust::device_ptr<const T> _a = input->data_gpu(), 
                                _b = M_weight.data_gpu(),
                                _bias = M_bias.data_gpu();
    thrust::device_ptr<T> _output = output->data_gpu();
    Linear_cuda<<<grid_size, block_size>>>
      (_a, _b, _bias, _output, icol,  ocol);
    cudaDeviceSynchronize();
    return output;
  }

  std::shared_ptr<Tensor<T>>
  forward_cpu(const std::shared_ptr<const Tensor<T>> input) {
    int icol = input->col(), ocol = M_bias.col(), num = input->number();
    auto output = std::make_shared<Tensor<T>>(1, ocol, 1, num, Device::CPU, 0);
    for(int n = 0; n < num; n++){
      int ioffset = n * icol;
      parallelizer.parallel_col(
        Linear_cpu<T>, output, ioffset, input, M_weight, M_bias);
    }
    parallelizer.sync();
    return output;
  }

  Tensor<T> M_weight, M_bias;
  Device m_device;
};

}