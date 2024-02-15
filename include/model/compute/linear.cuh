#pragma once

#include <basic/function.cuh>

namespace dl{

template<typename T=f32>
class Linear : public Function<T> {
public:
  explicit 
  Linear(int input_dim, int output_dim, Device device=Device::CPU) :
    M_weight(output_dim, input_dim, 1, 1, -1, device), 
    M_bias(1, output_dim, 1, 1, -1, device) { } 
  
  virtual ~Linear() = default;

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> input) override{
    auto mat = M_weight * (*input);
    mat = mat->sum();
    auto output = *mat + M_bias; 
    return output;
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<Tensor<T>> input){
    return forward(input);
  }

  void to(Device device){
    M_weight.to(device);
    M_bias.to(device);
    m_device = device;
  }

private:
  Tensor<T> M_weight, M_bias;
  Device m_device;
};
}