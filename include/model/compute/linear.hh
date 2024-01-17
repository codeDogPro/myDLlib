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
  Linear(int input_dim, int output_dim){
    M_weight = Tensor<T>(output_dim, input_dim, 1);
    M_bias = Tensor<T>(output_dim, 1);
  } 

  virtual ~Linear(){ };

  virtual Tensor<T>
  forward(const Tensor<T>& input) override{

    if(input.row() != 1) input.reshape(1, input.col(), input.channel());
    Tensor<T> mat = M_weight * input;
    return mat.sum() + M_bias; 
  }

  // virtual Tensor<T>
  // backward() override{
  //   return 
  // }


private:
  Tensor<T> M_weight, M_bias;
};
}