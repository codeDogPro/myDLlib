#pragma once

#include <data/tensor.hh>
#include <basic/function.hh>

namespace dl{

template<typename T=f32>
class Linear : public Function<T> {
public:
  explicit Linear(int input_dim, int output_dim) :
    M_weight(output_dim, input_dim), M_bias(1, output_dim) { } 
  
  virtual ~Linear() = default;

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> input) override{
    auto mat = M_weight * (*input);
    mat = mat->sum();
    auto output = *mat + M_bias; 
    return output;
  }

private:
  Tensor<T> M_weight, M_bias;
};
}