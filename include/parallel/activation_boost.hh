#pragma once

#include <basic/tensor_macro.hh>
#include <cmath>

namespace dl{

  template<typename T> class Tensor;

  template<typename Fn_ch, typename Fn_col, typename T>
  static void 
  activation_forward(Fn_ch&& f_ch, Fn_col&& f_col, Tensor<T>& res, const Tensor<T>& input){
    int row = input.row(), col = input.col(), channel = input.channel();
    int number = input.number(), volume = row * col * channel;
    int ncpu = cpu_number();

    for(int i = 0; i < number; i++){
      int noffset = volume * i;
      if(channel >= ncpu * BOOST_CHANNEL){
        puts("In parallel channel");
        parallel_channel(std::forward<Fn_ch>(f_ch), 
        /*nthread, res */NTHREAD_C(ncpu, number), res,
        /*const args...*/input, noffset);
      }
      else if(col >= ncpu * BOOST_COL){
        puts("In parallel col");
        parallel_col    (std::forward<Fn_col>(f_col), 
        /*nthread, res */NTHREAD_R(ncpu, number), res,
        /*const args...*/input, noffset);
      }
      else{ // No need to boost
        puts("No need to boost");
        f_ch(0, channel, 0, res, 
             input, noffset); 
      }
    }
  }

  template<typename T>
  inline int 
  relu_channel
  (int ch_begin, int ch_num, int pad, Tensor<T>& res, 
   const Tensor<T>& input, int noffset){
    int square = input.row() * input.col();
    int start = noffset + square * ch_begin, end = start + square * ch_num;
    for(int i = start; i < end; i++){
      res[i] = input[i] > 0 ? input[i] : 0;
    }
    return ch_begin;
  }

  template<typename T>
  inline int 
  relu_col
  (int col_begin, int col_num, int channel, Tensor<T>& res,
   const Tensor<T>& input, int noffset){
    int row = input.row(), col = input.col(), square = row * col;
    int start = noffset + square * channel + col * col_begin;
    int end = start + col * col_num;
    for(int i = start; i < end; i++){
      res[i] = input[i] > 0 ? input[i] : 0;
    }
    return col_begin;
  }

  template<typename T>
  inline int 
  sigmoid_col
  (int col_begin, int col_num, int channel, Tensor<T>& res,
   const Tensor<T>& input, int noffset){
    int row = input.row(), col = input.col(), square = row * col;
    int start = noffset + square * channel + col * col_begin;
    int end = start + col * col_num;
    for(int i = start; i < end; i++){
      res[i] = 1 / (1 + std::exp(-input[i])) + eps;
    }
    return col_begin;
  }
}