#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <assert.h>

#include <iostream>

namespace dl{

template<typename T>
class Tensor{

public:
  Tensor() = delete;

  Tensor(int row, int col){
    this->m_data  = std::move(std::vector<T>(row * col));
    this->m_shape = std::move(std::vector<int>({row, col}));
  }

  Tensor(int row, int col, int channel){
    this->m_data  = std::move(std::vector<T>(row * col * channel));
    this->m_shape = std::move(std::vector<int>({row, col, channel}));
  }

  Tensor(const std::vector<int> &shape){
    int product = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    printf("product: %d\n", product);  
    this->m_data  = std::move(std::vector<T>(product));
    this->m_shape = std::move(shape);
  }

  // deep copy
  Tensor(const Tensor<T> &t){ 
    this->m_data  = std::move(std::vector<T>(t.m_data.size()));
    this->m_shape = t.m_shape;
  }

  // move copy
  Tensor(Tensor<T> &&t){ 
    this->m_data  = std::move(t.m_data);
    this->m_shape = std::move(t.m_shape);
  }

  Tensor<T> &
  operator=(Tensor<T> &t){
    this->m_data  = std::move(std::vector<T>(t.m_data.size()));
    this->m_shape = t.m_shape;
    return this;
  }

  friend Tensor<T> &
  operator+(const Tensor<T> &a, const Tensor<T> &b);
  
  friend Tensor<T> &
  operator-(const Tensor<T> &a, const Tensor<T> &b);

  friend Tensor<T> &
  operator*(const Tensor<T> &a, const Tensor<T> &b);

  friend Tensor<T> &
  operator/(const Tensor<T> &a, const Tensor<T> &b);


  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel

private:
  std::vector<T> m_data;
};

}


#endif