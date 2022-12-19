#ifndef TENSOR_H
#define TENSOR_H

#include <data/rand_init.h>
#include <basic/enumaration.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>

#include <iostream>
#include <assert.h>

namespace dl{


template<typename T>
class Tensor{

public:
  Tensor() = delete;

  Tensor(int row, int col){
    this->m_data.assign(row * col, 0);
    this->m_shape.assign({row, col});
    rand_init(*this);
  }

  Tensor(int row, int col, int channel){
    this->m_data.assign(row * col * channel, 0);
    this->m_shape.assign({row, col, channel});
    rand_init(*this);
  }

  Tensor(const std::vector<int> &shape){
    int product = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    // printf("product: %d\n", product);  
    this->m_data.assign(product, 0);
    this->m_shape = std::move(shape);
    rand_init(*this);
  }

  // deep copy
  Tensor(const Tensor<T> &t){ 
    this->m_data.assign(t.m_data.size(), 0);
    this->m_shape = t.m_shape;
    rand_init(*this);
  }

  // move copy
  Tensor(Tensor<T> &&t){ 
    this->m_data  = std::move(t.m_data);
    this->m_shape = std::move(t.m_shape);
  }

  Tensor<T> &
  operator=(const Tensor<T> &t){
    this->m_data.assign(t.m_data.size(), 0);
    this->m_shape = t.m_shape;
    rand_init(*this);
    return *this;
  }

  Tensor<T> &
  operator=(Tensor<T> &&t){
    this->m_data  = std::move(t.m_data);
    this->m_shape = std::move(t.m_shape);
    return *this;
  }

  Tensor<T> 
  calculator(Tensor<T> &a, Tensor<T> &b, int mode);

  Tensor<T> 
  operator+(Tensor<T> &t){
    return calculator(*this, t, PLUS);
  }

  Tensor<T> 
  operator-(Tensor<T> &t){
    return calculator(*this, t, MINUS);
  }

  Tensor<T> 
  operator*(Tensor<T> &t){
    return calculator(*this, t, MULTIPLY);
  }

  Tensor<T> 
  operator/(Tensor<T> &t){
    return calculator(*this, t, DIVIDE);
  }

  T&
  operator[](int idx){
    return this->m_data[idx];
  }


  template<typename U>
  friend std::ostream &
  operator<<(std::ostream &os, const Tensor<U> &t);


  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel
  
  std::vector<T> &
  get_data(){return this->m_data;}

private:
  std::vector<T> m_data;
};


  template<typename U>
  std::ostream &
  operator<<(std::ostream &os, const Tensor<U> &t){
    int height = t.m_shape[0], width = t.m_shape[1];
    switch(t.m_shape.size()){
    case 2:
      printf("[");
      for(int h = 0; h < height; h++){
        int row_idx = h * width;
        if(h != 0)          putchar(' ');
        printf("[");
        for(int w = 0; w < width; w++){
          os << t.m_data[row_idx + w];
          if(w != width - 1) os << ", ";
        }
        printf("]");
        if(h != height - 1) putchar('\n');
      }
      printf("]\n"); break;
    case 3:
      int channel = t.m_shape[2];
      printf("[");
      for(int c = 0; c < channel; c++){
        printf("[");
        for(int h = 0; h < height; h++){
          int row_idx = h * width;
          printf("[");
          for(int w = 0; w < width; w++){
            os << t.m_data[row_idx + w];
            if(w != width - 1) os << ", ";
          }
          printf("]");
          if(h != height - 1) putchar('\n');
        }
        printf("]");
        if(c != channel - 1) putchar('\n');
      }
      printf("]\n"); break;
    }
    puts("");
    return os;
  }


}


#endif