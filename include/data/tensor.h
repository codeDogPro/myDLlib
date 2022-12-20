#ifndef TENSOR_H
#define TENSOR_H

#include <data/rand_init.h>
#include <data/thread_calcu.h>
#include <basic/enumaration.h>
#include <basic/timer.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <cstdlib>

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
    this->m_shape = shape;
    rand_init(*this);
  }

  Tensor(std::vector<int> &data, std::vector<int> &shape)
  : m_data(data), m_shape(shape){ }

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
    // puts("invoke +");
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
  
  void shape(){
    printf("shape:[");
    for(int i = 0; i < this->m_shape.size(); i++) {
      std::cout << this->m_shape[i];
      if(i != this->m_shape.size() - 1) printf(", ");
      else printf("]\n");
    }
  }

  std::vector<T> &
  get_data(){return this->m_data;}

private:
  std::vector<T> m_data;
};

//################### Tensor::member functions' implementation ###################

  template<typename T>
  Tensor<T> 
  Tensor<T>::calculator(Tensor<T> &a, Tensor<T> &b, int mode){
    assert(a.m_shape[1] == b.m_shape[1]);

    Tensor<T> result(a.m_shape);
    int ncpu = std::thread::hardware_concurrency();
#ifdef BENCH
    Timer t;
#endif
    // if the tensor is big enough, boost it.
    if(a.m_shape[0] >= ncpu * BOOST_THRD){ 
      int row_num = a.m_shape[0] / ncpu, col = a.m_shape[1];

      std::vector<std::thread> pool;
      for(int i = 0; i < ncpu; i++){
        int row_begin = i * row_num;
        if(a.m_shape[0] != b.m_shape[0]){
          if(b.m_shape[0] != 1) goto erro; 

          // puts("ready create thread");
          std::thread task(vec_single<T>, std::ref(a), std::ref(b), std::ref(result),
                          row_begin, row_num, col, mode);
          pool.push_back(std::move(task));
        } else{
          std::thread task(vec_full<T>, std::ref(a), std::ref(b), std::ref(result),
                          row_begin, row_num, col, mode);
          pool.push_back(std::move(task));
        }
      }
      for(auto &task : pool) task.join();
    } 
    else{ // don't need to boost.
      int row_num = a.m_shape[0], col = a.m_shape[1];
      if(a.m_shape[0] != b.m_shape[0]){
        if(b.m_shape[0] != 1) goto erro; 

        vec_single<T>(a, b, result, 0, row_num, col, mode);
      } else{
        vec_full  <T>(a, b, result, 0, row_num, col, mode);
      }
    }

    return result;

  erro:
    fprintf(stderr,
    "The size of tensor a:(%d) must match the size of tensor b:(%d) \
    at non-singleton dimension 0\n", a.m_shape[0], b.m_shape[0]);
    exit(-1);
  }



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