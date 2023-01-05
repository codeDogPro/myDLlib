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

    virtual Tensor<T>
    forward(Tensor<T> &input) override{
      if(auto_grad) grad = input;

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


  template<typename T>
  class Conv1D : public Function<T> {

  };


  template<typename T>
  class Conv2D : public Function<T> {
  public:
    Conv2D(int size, int stride=1, bool paddle=false, bool auto_grad=false){
      m_parameter = Tensor<T>(size, size, 1);
      m_stride = stride;
      m_paddle = paddle;
      m_auto_grad = auto_grad;
    }
  
    virtual Tensor<T>
    forward(){

    }


    int stride() { return m_stride; }
    bool paddle(){ return m_paddle; }

  private:
    bool m_auto_grad;
    bool m_paddle;
    int m_stride;
    Tensor<T> m_parameter, grad;
  };
}