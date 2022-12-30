#pragma once

#include <basic/enumaration.hpp>
#include <vector>
#include <iostream>
#include <assert.h>

// #define DEBUG

namespace dl{
  template<typename T> class Tensor;

//##################### Thread functions ########################

  template<typename T>
  void vec_channel_s
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &res, int ch_begin, int ch_num, int mode){
    int arow = a.m_shape[0], brow = b.m_shape[0], col = a.m_shape[1];
    int asquare = arow * col, bsquare = brow * col;
    for(int ch = ch_begin; ch < ch_begin + ch_num; ch++){
      int astart = ch * asquare, bstart = ch * bsquare, aend = astart + asquare;
      for(int i = astart; i < aend; i++){
        switch(mode){
          case PLUS:
          res[i] = a[i] + b[bstart + i % col]; break;
          case MINUS:
          res[i] = a[i] - b[bstart + i % col]; break;
          case MULTIPLY:
          res[i] = a[i] * b[bstart + i % col]; break;
          case DIVIDE:
          assert(b[bstart + i % col] != 0);
          res[i] = a[i] / b[bstart + i % col]; break;
          case MOD:
          res[i] = a[i] % b[bstart + i % col]; break;
          default: assert(0);
        }
      }
    }
  }

  template<typename T>
  void vec_channel_f
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &res, int ch_begin, int ch_num, int mode){
    int square = a.m_shape[0] * a.m_shape[1];
    int start = ch_begin * square, end = (ch_begin + ch_num) * square;
    for(int i = start; i < end; i++){
      switch(mode){
        case PLUS:
        res[i] = a[i] + b[i]; break;
        case MINUS:
        res[i] = a[i] - b[i]; break;
        case MULTIPLY:
        res[i] = a[i] * b[i]; break;
        case DIVIDE:
        assert(b[i] != 0);
        res[i] = a[i] / b[i]; break;
        case MOD:
        res[i] = a[i] % b[i]; break;
        default: assert(0);
      }
    }
  }

  template<typename T>
  void vec_row_s // the row number of a must >= b
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &res,
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
        res[i] = a[i] + b[bstart + i % col]; break;
        case MINUS:
        res[i] = a[i] - b[bstart + i % col]; break;
        case MULTIPLY:
        res[i] = a[i] * b[bstart + i % col]; break;
        case DIVIDE:
        assert(b[bstart + i % col] != 0);
        res[i] = a[i] / b[bstart + i % col]; break;
        case MOD:
        res[i] = a[i] % b[bstart + i % col]; break;
        default: assert(0);
      }
    }
#ifdef DEBUG
        printf("a:%d  b:%d  res:%d\n", a[col_idx], b[c], res[col_idx]);
#endif
  }
  
  template<typename T>
  void vec_row_f // a and b's shape must be the same
  (Tensor<T> &a, Tensor<T> &b, Tensor<T> &res,
  int channel, int row_begin, int row_num, int mode){
    int row = a.m_shape[0], col = a.m_shape[1], square = row * col;
    int start = channel * square + row_begin * col, end = start + row_num * col;
    for(int i = start; i < end; i++){
      switch(mode){
        case PLUS:
        res[i] = a[i] + b[i]; break;
        case MINUS:
        res[i] = a[i] - b[i]; break;
        case MULTIPLY:
        res[i] = a[i] * b[i]; break;
        case DIVIDE:
        assert(b[i] != 0);
        res[i] = a[i] / b[i]; break;
        case MOD:
        res[i] = a[i] % b[i]; break;
        default: assert(0);
      }
#ifdef DEBUG
        printf("a:%d  b:%d  res:%d\n", a[col_idx], b[col_idx], res[col_idx]);
#endif
    }
  }

  template<typename T>
  void sum_mean_axis0
  (Tensor<T> &t, Tensor<T> &res, int start, int end, int res_i, int mode){
    int col = t.m_shape[1]; 
    for(int i = start, cnt = 0, sum = 0; i < end; i++){
      sum += t[i];
      if(++cnt == col){
        if(mode == SUM)
          res[res_i++] = sum;
        if(mode == MEAN)
          res[res_i++] = sum / col;
        sum = cnt = 0;
      }
    }
  }

  template<typename T>
  void sum_mean_axis1_channel
  (Tensor<T> &t, Tensor<T> &res, int start, int end, int res_i, int mode){
    int col = t.m_shape[1], row = t.m_shape[0], square = row * col; 
    std::vector<T> sums(col, 0);
    for(int i = start, cnt = 0; i < end; i++){
      sums[i % col] += t[i];
      if(++cnt == square){
        for(auto &sum : sums){
          if(mode == SUM)
            res[res_i++] = sum;
          if(mode == MEAN)
            res[res_i++] = sum / row;
          sum = 0;
        }
        cnt = 0;
      } 
    }
  }

  template<typename T>
  void sum_mean_axis1_col
  (Tensor<T> &t, Tensor<T> &res, int start, int col_num, int res_i, int mode){
    int row = t.m_shape[0], col = t.m_shape[1], square = row * col_num;
    std::vector<T> sums(col_num, 0);
    for(int r = 0, i = start; r < row; r++, i += col)
      for(int c = 0; c < col_num; c++, i++)
        sums[c] += t[i];
    for(auto &sum : sums){
      if(mode == SUM)
        res[res_i++] = sum;
      if(mode == MEAN)
        res[res_i++] = sum / row;
    }
  }

  template<typename T>
  void sum_mean_axis2_row
  (Tensor<T> &t, Tensor<T> &res, int start, int end, int row_num, int res_i, int mode){
    int row = t.m_shape[0], col = t.m_shape[1];
    int zone = row_num * col, square = row * col;
    std::vector<T> sums(zone, 0);
    for(int i = start, cnt = 0; i < end; i++){
      sums[cnt] += t[i];
      if(++cnt == zone){
        i += square - zone;
        cnt = 0;
      }
    }
    for(auto & sum : sums){
      if(mode == SUM)
        res[res_i++] = sum;
      if(mode == MEAN)
        res[res_i++] = sum / row;
    }
  }

  template<typename T>
  void sum_mean_axis2_col
  (Tensor<T> &t, Tensor<T> &res, int start, int end, int col_num, int res_i, int mode){
    int row = t.m_shape[0], col = t.m_shape[1], channel = t.m_shape[2];
    int zone = col_num * row, square = row * col;
    std::vector<T> sums(zone, 0);
    for(int i = start, cnt = 0, sums_i = 0; i < end; i++){
      sums[sums_i++] += t[i];
      if(++cnt == col_num){
        i += col - col_num;
        cnt = 0;
      }
    }
    for(int cnt = 0; auto & sum : sums){
      if(mode == SUM)
        res[res_i++] = sum;
      if(mode == MEAN)
        res[res_i++] = sum / channel; 
      if(++cnt == col_num){
        res_i += col - col_num;
        cnt = 0;
      }
    }
  }
}
