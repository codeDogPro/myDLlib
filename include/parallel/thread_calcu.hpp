#pragma once

#include <basic/enumaration.hpp>
#include <iostream>
#include <assert.h>

// #define DEBUG

namespace dl{
  template<typename T> class Tensor;

//##################### Thread functions ########################

  template<typename T>
  void vec_channel_s
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result, int ch_begin, int ch_num, int mode){
    int arow = a.m_shape[0], brow = b.m_shape[0], col = a.m_shape[1];
    int asquare = arow * col, bsquare = brow * col;
    for(int ch = ch_begin; ch < ch_begin + ch_num; ch++){
      int astart = ch * asquare, bstart = ch * bsquare, aend = astart + asquare;
      for(int i = astart; i < aend; i++){
        switch(mode){
          case PLUS:
          result[i] = a[i] + b[bstart + i % col]; break;
          case MINUS:
          result[i] = a[i] - b[bstart + i % col]; break;
          case MULTIPLY:
          result[i] = a[i] * b[bstart + i % col]; break;
          case DIVIDE:
          assert(b[bstart + i % col] != 0);
          result[i] = a[i] / b[bstart + i % col]; break;
          case MOD:
          result[i] = a[i] % b[bstart + i % col]; break;
          default: assert(0);
        }
      }
    }
  }

  template<typename T>
  void vec_channel_f
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result, int ch_begin, int ch_num, int mode){
    int square = a.m_shape[0] * a.m_shape[1];
    int start = ch_begin * square, end = (ch_begin + ch_num) * square;
    for(int i = start; i < end; i++){
      switch(mode){
        case PLUS:
        result[i] = a[i] + b[i]; break;
        case MINUS:
        result[i] = a[i] - b[i]; break;
        case MULTIPLY:
        result[i] = a[i] * b[i]; break;
        case DIVIDE:
        assert(b[i] != 0);
        result[i] = a[i] / b[i]; break;
        case MOD:
        result[i] = a[i] % b[i]; break;
        default: assert(0);
      }
    }
  }

  template<typename T>
  void vec_row_s // the row number of a must >= b
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result,
   int channel, int row_begin, int row_num, int mode){
#ifdef DEBUG
    printf("row_num:%d col:%d\n", row_num, col);
#endif
    int arow = a.m_shape[0], brow = b.m_shape[0], col = a.m_shape[1];
    int asquare = arow * col, bsquare = brow * col;
    int astart = channel * asquare + row_begin * col, aend = astart + row_num * col;
    int bstart = channel * bsquare;
    for(int i = astart; i < aend; i++){
      switch(mode){
        case PLUS:
        result[i] = a[i] + b[bstart + i % col]; break;
        case MINUS:
        result[i] = a[i] - b[bstart + i % col]; break;
        case MULTIPLY:
        result[i] = a[i] * b[bstart + i % col]; break;
        case DIVIDE:
        assert(b[bstart + i % col] != 0);
        result[i] = a[i] / b[bstart + i % col]; break;
        case MOD:
        result[i] = a[i] % b[bstart + i % col]; break;
        default: assert(0);
      }
    }
#ifdef DEBUG
        printf("a:%d  b:%d  result:%d\n", a[col_idx], b[c], result[col_idx]);
#endif
  }
  
  template<typename T>
  void vec_row_f // a and b's shape must be the same
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &result,
  int channel, int row_begin, int row_num, int mode){
    int row = a.m_shape[0], col = a.m_shape[1], square = row * col;
    int start = channel * square + row_begin * col, end = start + row_num * col;
    for(int i = start; i < end; i++){
      switch(mode){
        case PLUS:
        result[i] = a[i] + b[i]; break;
        case MINUS:
        result[i] = a[i] - b[i]; break;
        case MULTIPLY:
        result[i] = a[i] * b[i]; break;
        case DIVIDE:
        assert(b[i] != 0);
        result[i] = a[i] / b[i]; break;
        case MOD:
        result[i] = a[i] % b[i]; break;
        default: assert(0);
      }
#ifdef DEBUG
        printf("a:%d  b:%d  result:%d\n", a[col_idx], b[col_idx], result[col_idx]);
#endif
    }
  }

}
