#pragma once

#include <basic/function.cuh>
#include <basic/tensor_macro.cuh>

#include <parallel/parallel.cuh>
#include <parallel/linear_cpu.cuh>
#include <parallel/linear_cuda.cuh>

namespace dl{

template<typename T=f32>
class Linear : public Function<T> {
public:
  explicit 
  Linear(int input_dim, int output_dim, Device device=Device::CPU) :
    M_weight(output_dim, input_dim, 1, 1, device), 
    M_bias(1, output_dim, 1, 1, device) { } 
  
  virtual ~Linear() = default;

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<const T>> input) override{
    return (input->device() == Device::CPU) ? forward_cpu(input) : forward_cuda(input);
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<Tensor<const T>> input){
    return forward(input);
  }

  void to(Device device){
    M_weight.to(device);
    M_bias.to(device);
    m_device = device;
  }

private:
  std::shared_ptr<Tensor<T>>
  forward_cuda(const std::shared_ptr<Tensor<const T>> input) {
    // TODO: implement it
    return nullptr;
  }

  std::shared_ptr<Tensor<T>>
  forward_cpu(const std::shared_ptr<Tensor<const T>> input) {
    int icol = input->col(), ocol = M_bias.col(), num = input->number();
    auto output = std::make_shared<Tensor<T>>(1, ocol, 1, num, Device::CPU, 0);
    for(int n = 0; n < num; n++){
      int offset = n * icol;
      parallelizer.parallel_col(
        Linear_cpu<T>, output,  offset,
        M_weight, input, M_bias);
    }
    parallelizer.sync();
    // auto mat = M_weight * (*input);
    // mat = mat->sum(0, false);
    return output;
  }

  Tensor<T> M_weight, M_bias;
  Device m_device;
};
}