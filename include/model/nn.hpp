#pragma once

#include <data/tensor.hpp>
#include <basic/function.hpp>

namespace dl{

  /*
  example:
    >>> std::vector<Function<int> *> model;
    >>> model.push_back(new Linear<int>(3, 2));
    >>> Tensor<int> input(1, 3, 1, 2);
    >>> auto output = model[0]->forward(input);
  */
  template<typename T>
  class Linear : public Function<T> {
  public:
    Linear(int input_dim, int output_dim, bool auto_grad=true){
      m_paremeter = Tensor<T>(output_dim, input_dim, 1, 3);
      auto_grad = auto_grad;
    } 

    virtual Tensor<T>
    forward(Tensor<T> &input) override{
      if(auto_grad) grad = input;

      std::cout << m_paremeter;
      Tensor<T> mat = m_paremeter * input;
      std::cout << mat;
      return mat.sum(); 
    }

    // virtual Tensor<T>
    // backward() override{
    //   return 
    // }

  private:
    bool auto_grad;
    Tensor<T> m_paremeter, grad;
  };

}