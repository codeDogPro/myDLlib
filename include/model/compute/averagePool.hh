#pragma once

#include <basic/function.hh>
#include <data/tensor.hh>
#include <parallel/pooling_parallel.hh>

namespace dl{

template<typename T>
class AveragePool2D : public Function<T> {
public:
  explicit
  AveragePool2D() = default;

  explicit
  AveragePool2D(int pooling_size=2, bool auto_grad=false) 
  : M_pool_size(pooling_size), M_auto_grad(auto_grad) {}

  virtual Tensor<T>
  forward(const Tensor<T>& input){
    int res_row = input.row() / M_pool_size, res_col = input.col() / M_pool_size;
    Tensor<T> res(res_row, res_col, input.channel(), 0, input.number());
    if(M_auto_grad) M_grad = input;

    pooling_forward(pool_channel<T>, pool_row<T>, res, input, Pool::AVG);

    return res;
  }
  
private:
  Tensor<T> M_grad;
  int M_pool_size;
  bool M_auto_grad;
};

}