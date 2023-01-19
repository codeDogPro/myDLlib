#pragma once

#include <basic/tensor_macro.hh>
#include <cmath>

namespace dl{

  template<typename T> class Tensor;

  template<typename Fn_ch, typename Fn_row, typename T>
  static void 
  activation_forward(Fn_ch&& f_ch, Fn_row&& f_row, Tensor<T>& res, const Tensor<T>& input){
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
      else if(row >= ncpu * BOOST_ROW){
        puts("In parallel row");
        parallel_row    (std::forward<Fn_row>(f_row), 
        /*nthread, res */NTHREAD_R(ncpu, number), res,
        /*const args...*/input, noffset);
      }
      else{ // No need to boost
        f_ch(0, channel, res, 
             input, noffset); 
      }
    }
  }

  template<typename T>
  inline int 
  relu_channel
  (int ch_begin, int ch_num, Tensor<T>& res, const Tensor<T>& input, int noffset){
    int square = input.row() * input.col();
    int start = noffset + square * ch_begin, end = start + square * ch_num;
    for(int i = start; i < end; i++){
      res[i] = input[i] > 0 ? input[i] : 0;
    }
    return ch_begin;
  }

  template<typename T>
  inline int 
  relu_row
  (int row_begin, int row_num, int channel, Tensor<T>& res,
   const Tensor<T>& input, int noffset){
    int row = input.row(), col = input.col(), square = row * col;
    int start = noffset + square * channel + col * row_begin;
    int end = start + col * row_num;
    for(int i = start; i < end; i++){
      res[i] = input[i] > 0 ? input[i] : 0;
    }
    return row_begin;
  }

  template<typename T>
  inline int 
  sigmoid_channel
  (int ch_begin, int ch_num, Tensor<T>& res, const Tensor<T>& input, int noffset){
    int square = input.row() * input.col();
    int start = noffset + square * ch_begin, end = start + square * ch_num;
    for(int i = start; i < end; i++){
      res[i] = input[i] > 0 ? input[i] : 0;
    }
    return ch_begin;
  }

  template<typename T>
  inline int 
  sigmoid_row
  (int row_begin, int row_num, int channel, Tensor<T>& res,
   const Tensor<T>& input, int noffset){
    int row = input.row(), col = input.col(), square = row * col;
    int start = noffset + square * channel + col * row_begin;
    int end = start + col * row_num;
    for(int i = start; i < end; i++){
      res[i] = input[i] > 0 ? input[i] : 0;
    }
    return row_begin;
  }
}