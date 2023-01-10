#pragma once

#include <data/tensor.hh>
#include <basic/function.hh>

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
      m_paremeter = Tensor<T>(output_dim, input_dim, 1);
      m_auto_grad = auto_grad;
    } 

    virtual ~Linear(){ };

    virtual Tensor<T>
    forward(Tensor<T> &input) override{
      if(m_auto_grad) grad = input;

      Tensor<T> mat = m_paremeter * input;
      return mat.sum(); 
    }

    // virtual Tensor<T>
    // backward() override{
    //   return 
    // }


  private:
    bool m_auto_grad;
    Tensor<T> m_paremeter, grad;
  };
}