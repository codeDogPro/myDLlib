#pragma once

#include <basic/tensor_macro.hh>
#include <vector>
#include <iostream>
#include <assert.h>

// #define DEBUG

namespace dl{
  template<typename T> class Tensor;

//##################### Thread functions ########################

  template<typename T>
  void vec_channel_s
  (const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &res,
   int ch_begin, int ch_num, int mode)
  {
    int arow = a.row(), brow = b.row(), col = a.col();
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
  (const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &res,
   int ch_begin, int ch_num, int mode)
  {
    int square = a.row() * a.col();
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
  (const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &res, 
   int channel, int row_begin, int row_num, int mode)
  {
#ifdef DEBUG
    printf("row_num:%d col:%d\n", row_num, col);
#endif
    int arow = a.row(), brow = b.row(), col = a.col();
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
  (const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &res,
   int channel, int row_begin, int row_num, int mode)
  {
    int row = a.row(), col = a.col(), square = row * col;
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

  using std::max, std::min;

  template<typename T>
  void operator_axis0
  (const Tensor<T> &t, Tensor<T> &res, int start, int end, int res_i, int mode){
    int col = t.col(), cnt = 0; 
    if(mode == SUM)
      for(int i = start, sum = 0; i < end; i++){
        sum += t[i];
        if(++cnt == col){
          if(mode == SUM)
            res[res_i++] = sum;
          if(mode == MEAN)
            res[res_i++] = sum / col;
          sum = cnt = 0;
        }
      }
    if(mode == MAX)
      for(int i = start, max_ = 1e-8; i < end; i++){
        max_ = max(max_, t[i]);
        if(++cnt == col){
          res[res_i++] = max_;
          cnt = 0; max_ = 1e-8;
        }
      }
    if(mode == MIN)
      for(int i = start, min_ = 1e8; i < end; i++){
        min_ = min(min_, t[i]);
        if(++cnt == col){
          res[res_i++] = min_;
          cnt = 0; min_ = 1e8;
        }
      }
  }

  template<typename T>
  void operator_axis1_channel
  (const Tensor<T> &t, Tensor<T> &res, int start, int end, int res_i, int mode){
    int col = t.col(), row = t.row(), square = row * col, cnt = 0; 
    if(mode == SUM || mode == MEAN){
      std::vector<T> sums(col, 0);
      for(int i = start; i < end; i++){
        sums[i % col] += t[i];
        if(++cnt == square){
          for(auto &sum : sums){
            if(mode == SUM)
              res[res_i++] = sum;
            if(mode == MEAN)
              res[res_i++] = sum / row;
            sum = 0;
          } cnt = 0;
        } 
      }
    }
    if(mode == MAX){
      std::vector<T> maxs(col, 1e-8);
      for(int i = start; i < end; i++){
        maxs[i % col] = max(maxs[i % col], t[i]);
        if(++cnt == square){
          for(auto &max_ : maxs){
            res[res_i++] = max_; max_ = 1e-8;
          } cnt = 0;
        } 
      }
    }
    if(mode == MIN){
      std::vector<T> mins(col, 1e8);
      for(int i = start; i < end; i++){
        mins[i % col] = min(mins[i % col], t[i]);
        if(++cnt == square){
          for(auto &min_ : mins){
            res[res_i++] = min_; min_ = 1e8;
          } cnt = 0;
        } 
      }
    }
  }

  template<typename T>
  void operator_axis1_col
  (const Tensor<T> &t, Tensor<T> &res, int start, int col_num, int res_i, int mode){
    int row = t.row(), col = t.col(), square = row * col_num;
    if(mode == SUM || mode == MEAN){
      std::vector<T> sums(col_num, 0);
      for(int r = 0, i = start; r < row; r++, i += col - col_num)
        for(int c = 0; c < col_num; c++, i++)
          sums[c] += t[i];
      for(auto sum : sums){
        if(mode == SUM)
          res[res_i++] = sum;
        if(mode == MEAN)
          res[res_i++] = sum / row;
      }
    }
    if(mode == MAX){
      std::vector<T> maxs(col_num, 1e-8);
      for(int r = 0, i = start; r < row; r++, i += col - col_num)
        for(int c = 0; c < col_num; c++, i++)
          maxs[c] = max(maxs[c], t[i]);
      for(auto max : maxs) res[res_i++] = max;
    }
    if(mode == MIN){
      std::vector<T> mins(col_num, 1e8);
      for(int r = 0, i = start; r < row; r++, i += col - col_num)
        for(int c = 0; c < col_num; c++, i++)
          mins[c] = min(mins[c], t[i]);
      for(auto min : mins) res[res_i++] = min;
    }
  }

  template<typename T>
  void operator_axis2_row
  (const Tensor<T> &t, Tensor<T> &res, int start, int end, int row_num, int res_i, int mode){
    int row = t.row(), col = t.col();
    int zone = row_num * col, square = row * col, cnt = 0;
    if(mode == SUM || mode == MEAN){
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
    if(mode == MAX){
      std::vector<T> maxs(zone, 1e-8);
      for(int i = start; i < end; i++){
        maxs[cnt] = max(maxs[cnt], t[i]);
        if(++cnt == zone){
          i += square - zone;
          cnt = 0;
        }
      }
      for(auto max : maxs) res[res_i++] = max;
    }
    if(mode == MIN){
      std::vector<T> mins(zone, 1e8);
      for(int i = start; i < end; i++){
        mins[cnt] = min(mins[cnt], t[i]);
        if(++cnt == zone){
          i += square - zone;
          cnt = 0;
        }
      }
      for(auto min : mins) res[res_i++] = min;
    }
  }

  template<typename T>
  void operator_axis2_col
  (const Tensor<T> &t, Tensor<T> &res, int start, int end, int col_num, int res_i, int mode){
    int row = t.row(), col = t.col(), channel = t.channel();
    int zone = col_num * row, square = row * col;
    if(mode == SUM || mode == MEAN){
      std::vector<T> sums(zone, 0);
      for(int i = start, cnt = 0, sums_i = 0; i < end; i++, sums_i++){
        sums[sums_i % zone] += t[i];
        if(++cnt == col_num){
          i += col - col_num;
          cnt = 0;
        }
      }
      for(int cnt = 0; auto sum : sums){
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
    if(mode == MAX){
      std::vector<T> maxs(zone, 1e-8);
      for(int i = start, cnt = 0, maxs_i = 0; i < end; i++, maxs_i++){
        maxs[maxs_i % zone] = max(maxs[maxs_i % zone], t[i]);
        if(++cnt == col_num){
          i += col - col_num;
          cnt = 0;
        }
      }
      for(int cnt = 0; auto max : maxs){
        res[res_i++] = max;
        if(++cnt == col_num){
          res_i += col - col_num;
          cnt = 0;
        }
      }
    }
    if(mode == MIN){
      std::vector<T> mins(zone, 1e-8);
      for(int i = start, cnt = 0, mins_i = 0; i < end; i++, mins_i++){
        mins[mins_i % zone] = min(mins[mins_i % zone], t[i]);
        if(++cnt == col_num){
          i += col - col_num;
          cnt = 0;
        }
      }
      for(int cnt = 0; auto min : mins){
        res[res_i++] = min;
        if(++cnt == col_num){
          res_i += col - col_num;
          cnt = 0;
        }
      }
    }
  }
}
