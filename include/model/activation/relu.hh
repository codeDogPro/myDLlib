#pragma once

#include <basic/function.hh>
#include <data/tensor.hh>
#include <parallel/activation_boost.hh>

namespace dl{
  
template<typename T>
class Relu : public Function<T> {
public:
  explicit
  Relu() = delete;

  explicit
  Relu(bool auto_grad=false) : M_auto_grad(auto_grad) {}

  virtual Tensor<T> 
  forward(const Tensor<T>& input){
    Tensor<T> res(input.get_cshape(), 0);
    if(M_auto_grad) M_grad = input;

    activation_forward(relu_channel<T>, relu_col<T>, res, input);

    return res;
  }

private:
  bool M_auto_grad;
  Tensor<T> M_grad;
};

}