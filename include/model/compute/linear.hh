#pragma once

#include <data/tensor.hh>
#include <basic/function.hh>

namespace dl{

template<typename T=f32>
class Linear : public Function<T> {
public:
  Linear(int input_dim, int output_dim) :
    M_weight(output_dim, input_dim), M_bias(1, output_dim) { } 

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> input) override{
    if(input->row() != 1) input->reshape(1, input->col(), input->channel());
    // std::cout << "input: ";
    // input->shape();
    auto mat = M_weight * (*input);
    // std::cout << "mat: ";
    // mat->shape();
    mat = mat->sum();
    // std::cout << "mat_sum: ";
    // mat->shape();
    auto output = *mat + M_bias; 
    // std::cout << "output: ";
    // output->shape();
    return output;
  }

private:
  Tensor<T> M_weight, M_bias;
};
}