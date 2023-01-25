#pragma once

#include <basic/tensor_macro.hh>
#include <vector>
#include <iostream>
#include <assert.h>


namespace dl{
  template<typename T> class Tensor;

//##################### Thread functions ########################

  template<typename T>
  int vec_channel_s
  (int ch_begin, int ch_num, int pad, Tensor<T>& res,
   const Tensor<T>& a, const Tensor<T>& b, int noffset, Calculator mode) {
    int arow = a.row(), brow = b.row(), col = a.col();
    int asquare = arow * col, bsquare = brow * col;
    for(int ch = ch_begin; ch < ch_begin + ch_num; ch++){
      int astart = noffset + ch * asquare, bstart = noffset + ch * bsquare;
      int aend = astart + asquare;
      for(int i = astart; i < aend; i++){
        switch(mode){
          case Calculator::PLUS:
          res[i] = a[i] + b[bstart + i % col]; break;
          case Calculator::MINUS:
          res[i] = a[i] - b[bstart + i % col]; break;
          case Calculator::MULTIPLY:
          res[i] = a[i] * b[bstart + i % col]; break;
          case Calculator::DIVIDE:
          assert(b[bstart + i % col] != 0);
          res[i] = a[i] / b[bstart + i % col]; break;
          case Calculator::MOD:
          res[i] = a[i] % b[bstart + i % col]; break;
          default: assert(0);
        }
      }
    }
    return ch_begin;
  }

  template<typename T>
  int vec_channel_f
  (int ch_begin, int ch_num, int pad, Tensor<T>& res,
   const Tensor<T>& a, const Tensor<T>& b, int noffset, Calculator mode) {
    int square = a.row() * a.col();
    int start = noffset + ch_begin * square;
    int end   = noffset + start + ch_num * square;
    for(int i = start; i < end; i++){
      switch(mode){
        case Calculator::PLUS:
        res[i] = a[i] + b[i]; break;
        case Calculator::MINUS:
        res[i] = a[i] - b[i]; break;
        case Calculator::MULTIPLY:
        res[i] = a[i] * b[i]; break;
        case Calculator::DIVIDE:
        assert(b[i] != 0);
        res[i] = a[i] / b[i]; break;
        case Calculator::MOD:
        res[i] = a[i] % b[i]; break;
        default: assert(0);
      }
    }
    return ch_begin;
  }

  template<typename T>
  int vec_row_s // the row number of a must >= b
  (int row_begin, int row_num, int channel, Tensor<T>& res,
   const Tensor<T>& a, const Tensor<T>& b, int noffset, Calculator mode) {
    int arow = a.row(), brow = b.row(), col = a.col();
    int asquare = arow * col, bsquare = brow * col;
    int astart = noffset + channel * asquare + row_begin * col;
    int bstart = noffset + channel * bsquare;
    int aend = astart + row_num * col;
    for(int i = astart; i < aend; i++){
      switch(mode){
        case Calculator::PLUS:
        res[i] = a[i] + b[bstart + i % col]; break;
        case Calculator::MINUS:
        res[i] = a[i] - b[bstart + i % col]; break;
        case Calculator::MULTIPLY:
        res[i] = a[i] * b[bstart + i % col]; break;
        case Calculator::DIVIDE:
        assert(b[bstart + i % col] != 0);
        res[i] = a[i] / b[bstart + i % col]; break;
        case Calculator::MOD:
        res[i] = a[i] % b[bstart + i % col]; break;
        default: assert(0);
      }
    }
    return row_begin;
  }
  
  template<typename T>
  int vec_row_f // a and b's shape must be the same
  (int row_begin, int row_num, int channel, Tensor<T>& res,
   const Tensor<T>& a, const Tensor<T>& b, int noffset, Calculator mode) {
    int row = a.row(), col = a.col(), square = row * col;
    int start = noffset + channel * square + row_begin * col;
    int end   = start + row_num * col;
    for(int i = start; i < end; i++){
      switch(mode){
        case Calculator::PLUS:
        res[i] = a[i] + b[i]; break;
        case Calculator::MINUS:
        res[i] = a[i] - b[i]; break;
        case Calculator::MULTIPLY:
        res[i] = a[i] * b[i]; break;
        case Calculator::DIVIDE:
        assert(b[i] != 0);
        res[i] = a[i] / b[i]; break;
        case Calculator::MOD:
        res[i] = a[i] % b[i]; break;
        default: assert(0);
      }
    }
    return row_begin;
  }

  using std::max, std::min;

  template<typename T>
  int operator_axis0_channel
  (int ch_begin, int ch_num, int pad, Tensor<T>& res, 
   const Tensor<T>& t, int noffset, int roffset, Operator mode) {
    int row = t.row(), col = t.col(), square = row * col, cnt = 0; 
    int start = noffset + square * ch_num * ch_begin, end = start + square * ch_num;
    int res_i = roffset + ch_num * ch_begin * row;

    if(mode == Operator::SUM || mode == Operator::MEAN)
      for(int i = start, sum = 0; i < end; i++){
        sum += t[i];
        if(++cnt == col){
          if(mode == Operator::SUM)
            res[res_i++] = sum;
          if(mode == Operator::MEAN)
            res[res_i++] = sum / col;
          sum = cnt = 0;
        }
      }
    else if(mode == Operator::MAX)
      for(int i = start, max_ = 1e-8; i < end; i++){
        max_ = max(max_, t[i]);
        if(++cnt == col){
          res[res_i++] = max_;
          cnt = 0; max_ = 1e-8;
        }
      }
    else if(mode == Operator::MIN)
      for(int i = start, min_ = 1e8; i < end; i++){
        min_ = min(min_, t[i]);
        if(++cnt == col){
          res[res_i++] = min_;
          cnt = 0; min_ = 1e8;
        }
      }
    return ch_begin;
  }

  template<typename T>
  int operator_axis0_row
  (int row_begin, int row_num, int channel, Tensor<T>& res, 
   const Tensor<T>& t, int noffset, int roffset, Operator mode) {
    int row = t.row(), col = t.col(), square = row * col, cnt = 0; 
    int start = noffset + square * channel + row_num * row_begin * col;
    int end = start + col * row_num;
    int res_i = roffset + channel * row + row_num * row_begin;

    if(mode == Operator::SUM || mode == Operator::MEAN)
      for(int i = start, sum = 0; i < end; i++){
        sum += t[i];
        if(++cnt == col){
          if(mode == Operator::SUM)
            res[res_i++] = sum;
          if(mode == Operator::MEAN)
            res[res_i++] = sum / col;
          sum = cnt = 0;
        }
      }
    else if(mode == Operator::MAX)
      for(int i = start, max_ = 1e-8; i < end; i++){
        max_ = max(max_, t[i]);
        if(++cnt == col){
          res[res_i++] = max_;
          cnt = 0; max_ = 1e-8;
        }
      }
    else if(mode == Operator::MIN)
      for(int i = start, min_ = 1e8; i < end; i++){
        min_ = min(min_, t[i]);
        if(++cnt == col){
          res[res_i++] = min_;
          cnt = 0; min_ = 1e8;
        }
      }
    return row_begin;
  }

  template<typename T>
  int operator_axis1_channel
  // (const Tensor<T> &t, Tensor<T> &res, int start, int end, int res_i, auto mode){
  (int ch_begin, int ch_num, int pad, Tensor<T>& res, 
   const Tensor<T>& t, int noffset, int roffset, Operator mode) {
    int row = t.row(), col = t.col(), square = row * col, cnt = 0; 
    int start = noffset + square * ch_num * ch_begin, end = start + square * ch_num;
    int res_i = roffset + ch_num * ch_begin * col;

    if(mode == Operator::SUM || mode == Operator::MEAN){
      std::vector<T> sums(col, 0);
      for(int i = start; i < end; i++){
        sums[i % col] += t[i];
        if(++cnt == square){
          for(auto &sum : sums){
            if(mode == Operator::SUM)
              res[res_i++] = sum;
            if(mode == Operator::MEAN)
              res[res_i++] = sum / row;
            sum = 0;
          } cnt = 0;
        } 
      }
    }
    else if(mode == Operator::MAX){
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
    else if(mode == Operator::MIN){
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
    return 1;
  }

  template<typename T>
  int operator_axis1_col
  (int col_begin, int col_num, int channel, Tensor<T>& res, 
   const Tensor<T>& t, int noffset, int roffset, Operator mode){
    int row = t.row(), col = t.col(), square = row * col_num;
    int start = noffset + square * channel + col_num * col_begin;
    int res_i = roffset + channel * col + col_num * col_begin;

    if(mode == Operator::SUM || mode == Operator::MEAN){
      std::vector<T> sums(col_num, 0);
      for(int r = 0, i = start; r < row; r++, i += col - col_num)
        for(int c = 0; c < col_num; c++, i++)
          sums[c] += t[i];
      for(auto sum : sums){
        if(mode == Operator::SUM)
          res[res_i++] = sum;
        if(mode == Operator::MEAN)
          res[res_i++] = sum / row;
      }
    }
    else if(mode == Operator::MAX){
      std::vector<T> maxs(col_num, 1e-8);
      for(int r = 0, i = start; r < row; r++, i += col - col_num)
        for(int c = 0; c < col_num; c++, i++)
          maxs[c] = max(maxs[c], t[i]);
      for(auto max : maxs) res[res_i++] = max;
    }
    else if(mode == Operator::MIN){
      std::vector<T> mins(col_num, 1e8);
      for(int r = 0, i = start; r < row; r++, i += col - col_num)
        for(int c = 0; c < col_num; c++, i++)
          mins[c] = min(mins[c], t[i]);
      for(auto min : mins) res[res_i++] = min;
    }
    return col_begin;
  }

  template<typename T>
  int operator_axis2_row
  (int row_begin, int row_num, int nouse, Tensor<T>& res, 
   const Tensor<T>& t, int noffset, int roffset, Operator mode){
    int row = t.row(), col = t.col(), channel = t.channel();
    int zone = row_num * col, square = row * col, cnt = 0;
    int start = noffset + row_num * row_begin * col;
    int end = start + (channel - 1) * square + row_num * col;
    int res_i = roffset + start;

    if(mode == Operator::SUM || mode == Operator::MEAN){
      std::vector<T> sums(zone, 0);
      for(int i = start, cnt = 0; i < end; i++){
        sums[cnt] += t[i];
        if(++cnt == zone){
          i += square - zone;
          cnt = 0;
        }
      }
      for(auto & sum : sums){
        if(mode == Operator::SUM)
          res[res_i++] = sum;
        if(mode == Operator::MEAN)
          res[res_i++] = sum / row;
      } 
    }
    else if(mode == Operator::MAX){
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
    else if(mode == Operator::MIN){
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
    return row_begin;
  }

  template<typename T>
  int operator_axis2_col
  (int col_begin, int col_num, int pad, Tensor<T>& res, 
   const Tensor<T>& t, int noffset, int roffset, Operator mode){
    int row = t.row(), col = t.col(), channel = t.channel();
    int zone = col_num * row, square = row * col;
    int start = noffset + col_num * col_begin;
    int end = start + (channel - 1) * square + (row - 1) * col + col_num;
    int res_i = roffset + start;

    if(mode == Operator::SUM || mode == Operator::MEAN){
      std::vector<T> sums(zone, 0);
      for(int i = start, cnt = 0, sums_i = 0; i < end; i++, sums_i++){
        sums[sums_i % zone] += t[i];
        if(++cnt == col_num){
          i += col - col_num;
          cnt = 0;
        }
      }
      for(int cnt = 0; auto sum : sums){
        if(mode == Operator::SUM)
          res[res_i++] = sum;
        if(mode == Operator::MEAN)
          res[res_i++] = sum / channel; 
        if(++cnt == col_num){
          res_i += col - col_num;
          cnt = 0;
        }
      }
    }
    else if(mode == Operator::MAX){
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
    else if(mode == Operator::MIN){
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
    return col_begin;
  }

} // namespace dl
