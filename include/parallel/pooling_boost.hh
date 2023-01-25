#pragma once

#include <basic/tensor_macro.hh>
#include <vector>

namespace dl{

  template<typename T> class Tensor;

  template<typename Fn_ch, typename Fn_row, typename T>
  static void 
  pooling_forward(Fn_ch f_ch, Fn_row f_row, 
  Tensor<T>& res, const Tensor<T>& input, Pool mode){
    int row = input.row(), col = input.col(), channel = input.channel();
    int number = input.number(), volume = row * col * channel;
    size_t ncpu = cpu_number();
    for(int i = 0; i < number; i++){
      int noffset = volume * i;
      if(channel >= ncpu * BOOST_CHANNEL){
        parallel_channel(std::forward<Fn_ch>(f_ch), 
        /*nthread, res */NTHREAD_C(ncpu, number), res,
        /*const args...*/input, noffset, mode);
      }
      else if(row >= ncpu * BOOST_ROW){
        parallel_row    (std::forward<Fn_row>(f_row), 
        /*nthread, res */NTHREAD_R(ncpu, number), res,
        /*const args...*/input, noffset, mode);
      }
      else{
        f_ch(0, channel, res, 
             input, noffset, mode); 
      }
    }
  }

  template<typename T>
  static int 
  pool_channel
  (int ch_begin, int ch_num, int pad, Tensor<T>& res,
   const Tensor<T>& input, int noffset, Pool mode){
    int start = 0;
    if(mode == Pool::MAX){
      // for(int i = start;)
      std::vector<T> maxs(res.col());
      //TODO: implement it
    }
    else if(mode == Pool::AVG){
      //TODO: implement it
    }
    return ch_begin;
  }

  template<typename T>
  static int 
  pool_row
  (int row_begin, int row_num, int channel, Tensor<T>& res,
   const Tensor<T>& input, int noffset, Pool mode){
    if(mode == Pool::MAX){
      //TODO: implement it
    }
    else if(mode == Pool::AVG){
      //TODO: implement it
    }
    return row_begin;
  }

}