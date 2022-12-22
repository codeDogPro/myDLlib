#ifndef THREAD_CALCU_H
#define THREAD_CALCU_H

#include <basic/enumaration.h>
#include <iostream>
#include <assert.h>

// #define DEBUG

namespace dl{
  template<typename T> class Tensor;

//##################### Thread functions ########################

  template<typename T>
  void vec_single // the row number of a must >= b
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result,
    int channel_offset, int row_begin, int row_num, int col, int mode){
#ifdef DEBUG
    printf("row_num:%d col:%d\n", row_num, col);
#endif
    for(int r = 0; r < row_num; r++){
      int row_idx = channel_offset + (row_begin + r) * col;
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
          case MOD:
          result[col_idx] = a[col_idx] % b[c]; break;
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
    int channel_offset, int row_begin, int row_num, int col, int mode){
    for(int r = 0; r < row_num; r++){
      int row_idx = channel_offset + (row_begin + r) * col;
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
          case MOD:
          result[col_idx] = a[col_idx] % b[col_idx]; break;
          default: assert(0);
        }
#ifdef DEBUG
        printf("a:%d  b:%d  result:%d\n", a[col_idx], b[col_idx], result[col_idx]);
#endif
      }
    }
  }

  template<typename T>
  void vec_full_s // just add a.m_data and b.m_data
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result, int mode){
    for(int i = 0; i < a.get_data().size(); i++){
      switch(mode){
        case PLUS:
        result[i] = a[i] + b[i]; break;
        case MINUS:
        result[i] = a[i] - b[i]; break;
        case MULTIPLY:
        result[i] = a[i] * b[i]; break;
        case DIVIDE:
        result[i] = a[i] / b[i]; break;
        case MOD:
        result[i] = a[i] % b[i]; break;
        default: assert(0);
      }
    }
  }
}

#endif