#pragma once

namespace dl{

  template<typename T> class Tensor;

  template<typename T>
  void
  paddle(Tensor<T> &input, Tensor<T> &pad_input, int npaddle){
    int irow = input.row(), icol = input.col(), ichannel = input.channel();
    int prow = pad_input.row(), pcol = pad_input.col();
    int pad_i = npaddle * (pcol + 1);
    for(int c_cnt = 0, r_cnt = 0; auto x : input.get_data()){
      pad_input[pad_i++] = x;
      if(++c_cnt == icol){
        pad_i += 2 * npaddle;
        c_cnt = 0;
        if(++r_cnt == irow){
          pad_i += 2 * npaddle * pcol;
          r_cnt = 0;
        }
      } 
    }
  }
}