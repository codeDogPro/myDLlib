#ifndef THREAD_CALCU_H
#define THREAD_CALCU_H

#include <basic/enumaration.h>
#include <assert.h>

#include <iostream>

// #define DEBUG

namespace dl{
  template<typename T> class Tensor;

//##################### Thread functions ########################

  template<typename T>
  void vec_single // the row number of a must >= b
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result,
    int row_begin, int row_num, int col, int mode){
#ifdef DEBUG
    printf("row_num:%d col:%d\n", row_num, col);
#endif
    for(int r = 0; r < row_num; r++){
      int row_idx = (row_begin + r) * col;
      for(int c = 0; c < col; c++){
        int col_idx = row_idx + c;
        switch(mode){
          case PLUS:
          result[col_idx] = a[col_idx] + b[c]; break;
          case MINUS:
          result[col_idx] = a[col_idx] - b[c]; break;
          case MULTIPLY:
          result[col_idx] = a[col_idx] * b[c]; break;
          case DIVIDE:
          result[col_idx] = a[col_idx] / b[c]; break;
          default: assert(0);
        }
#ifdef DEBUG
        printf("a:%d  b:%d  result:%d\n", a[col_idx], b[c], result[col_idx]);
#endif
      }
    }
  }
  
  template<typename T>
  void vec_full // a and b's shape must be the same
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result,
    int row_begin, int row_num, int col, int mode){
    for(int r = 0; r < row_num; r++){
      int row_idx = (row_begin + r) * col;
      for(int c = 0; c < col; c++){
        int col_idx = row_idx + c;
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
#ifdef DEBUG
        printf("a:%d  b:%d  result:%d\n", a[col_idx], b[col_idx], result[col_idx]);
#endif
      }
    }
  }

}

#endif