#ifndef TENSOR_H
#define TENSOR_H

// #include <data/rand_init.h>
#include <basic/enumaration.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <assert.h>

#include <iostream>

namespace dl{

template<typename T>
class Tensor;

template<typename T>
Tensor<T> *
calculator(const Tensor<T> &a, const Tensor<T> &b, int mode);

template<typename T>
class Tensor{

public:
  Tensor() = delete;

  Tensor(int row, int col){
    this->m_data.assign(row * col, 0);
    this->m_shape.assign({row, col});
  }

  Tensor(int row, int col, int channel){
    this->m_data.assign(row * col * channel, 0);
    this->m_shape.assign({row, col, channel});
  }

  Tensor(const std::vector<int> &shape){
    int product = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    printf("product: %d\n", product);  
    this->m_data.assign(product, 0);
    this->m_shape = std::move(shape);
  }

  // deep copy
  Tensor(const Tensor<T> &t){ 
    this->m_data.assign(t.m_data.size(), 0);
    this->m_shape = t.m_shape;
  }

  // move copy
  Tensor(Tensor<T> &&t){ 
    this->m_data  = std::move(t.m_data);
    this->m_shape = std::move(t.m_shape);
  }

  Tensor<T> &
  operator=(Tensor<T> &t){
    this->m_data.assign(t.m_data.size(), 0);
    this->m_shape = t.m_shape;
    return this;
  }

  Tensor<T> *
  operator+(const Tensor<T> &b){
    return calculator(*this, b, PLUS);
  }

  Tensor<T> *
  operator-(const Tensor<T> &b){
    return calculator(*this, b, MINUS);
  }

  Tensor<T> *
  operator*(const Tensor<T> &b){
    return calculator(*this, b, MULTIPLY);
  }

  Tensor<T> *
  operator/(const Tensor<T> &b){
    return calculator(*this, b, DIVIDE);
  }

  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel

private:
  std::vector<T> m_data;
};

}


#endif