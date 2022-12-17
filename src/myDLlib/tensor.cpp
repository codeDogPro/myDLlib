#include <tensor.h>
#include <enumaration.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <future>

#include <numeric>
#include <algorithm>

#include <memory>

#include <assert.h>
#include <iostream>

namespace dl{

  /*
  TODO: add a abstract function to implement +-/* all in once
  */
  template<typename T>
  static Tensor<T> &
  calculator(const Tensor<T> &a, const Tensor<T> &b, int mode){
    assert(a.m_shape[1] == b.m_shape[1]);

    Tensor<T> result(a.m_shape);

    if(a.m_shape[0] != b.m_shape[0]){
      if(b.m_shape[0] == 1){
        // TODO:mode b to each row of a
        return result;
      }

      fprintf(stderr,
      "The size of tensor a:(%d) must match the size of tensor b:(%d) \
      at non-singleton dimension 0\n", a.m_shape[0], b.m_shape[0]);
      assert(0);
    }
    else{
      // TODO:invoke matrix mode()
    }
  }

  template<typename T>
  Tensor<T> &
  operator+(const Tensor<T> &a, const Tensor<T> &b){
    return calculator(a, b, PLUS);
  }

  template<typename T>
  Tensor<T> &
  operator-(const Tensor<T> &a, const Tensor<T> &b){
    return calculator(a, b, MINUS);
  }

  template<typename T>
  Tensor<T> &
  operator*(const Tensor<T> &a, const Tensor<T> &b){
    return calculator(a, b, MULTIPLY);
  }

  template<typename T>
  Tensor<T> &
  operator/(const Tensor<T> &a, const Tensor<T> &b){
    return calculator(a, b, DIVIDE);
  }

}