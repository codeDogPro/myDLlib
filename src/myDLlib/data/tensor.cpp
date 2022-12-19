#include <data/tensor.h>
#include <basic/enumaration.h>

#include <thread>
#include <cstdlib>

#include <numeric>
#include <algorithm>

#include <memory>

#include <assert.h>
#include <iostream>

namespace dl{

//##################### Thread functions ########################
  template<typename T>
  static void vec_single // the row number of a must >= b
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result,
    int row_begin, int row_num, int col, int mode){
    for(int i = 0; i < row_num; i++){
      int row_idx = (row_begin + i) * col;
      for(int j = 0; j < col; j++){
        int col_idx = row_idx + col;
        switch(mode){
          case PLUS:
          result[col_idx] = a[col_idx] + b[col]; break;
          case MINUS:
          result[col_idx] = a[col_idx] - b[col]; break;
          case MULTIPLY:
          result[col_idx] = a[col_idx] * b[col]; break;
          case DIVIDE:
          result[col_idx] = a[col_idx] / b[col]; break;
          default: assert(0);
        }
      }
    }
  }
  
  template<typename T>
  static void vec_full // a and b's shape must be the same
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result,
    int row_begin, int row_num, int col, int mode){
    for(int i = 0; i < row_num; i++){
      int row_idx = (row_begin + i) * col;
      for(int j = 0; j < col; j++){
        int col_idx = row_idx + col;
        switch(mode){
          case PLUS:
          result[col_idx] = a[col_idx] + b[col_idx]; break;
          case MINUS:
          result[col_idx] = a[col_idx] - b[col_idx]; break;
          case MULTIPLY:
          result[col_idx] = a[col_idx] * b[col_idx]; break;
          case DIVIDE:
          result[col_idx] = a[col_idx] / b[col_idx]; break;
          default: assert(0);
        }
      }
    }
  }


//################### Tensor::member functions' implementation ###################

  template<typename T>
  Tensor<T> 
  Tensor<T>::calculator(Tensor<T> &a, Tensor<T> &b, int mode){
    assert(a.m_shape[1] == b.m_shape[1]);

    Tensor<T> result(a.m_shape);

    int ncpu = std::thread::hardware_concurrency();
    int row_num = a.m_shape[0] / ncpu;
    // printf("ncpu: %d\n", ncpu);

    std::vector<std::thread> pool;
    for(int i = 0; i < ncpu; i++){
      int row_begin = i * row_num;
      if(a.m_shape[0] != b.m_shape[0]){
        if(b.m_shape[0] != 1) goto erro; 

        std::thread task(vec_single<T>, std::ref(a), std::ref(b), std::ref(result),
                         row_begin, row_num, a.m_shape[1], mode);
        pool.push_back(std::move(task));
      } else{
        std::thread task(vec_full<T>, std::ref(a), std::ref(b), std::ref(result),
                         row_begin, row_num, a.m_shape[1], mode);
        pool.push_back(std::move(task));
      }
    }
    for(auto &task : pool) task.join();

    return result;

  erro:
    fprintf(stderr,
    "The size of tensor a:(%d) must match the size of tensor b:(%d) \
    at non-singleton dimension 0\n", a.m_shape[0], b.m_shape[0]);
    exit(-1);
  }


  template class Tensor<int>;
  template class Tensor<float>;
  template class Tensor<double>;

}