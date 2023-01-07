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

  template<typename T>
  void conv2d_channel
  (Tensor<T> &input, Tensor<T> &kernel, Tensor<T> &res, int ch_begin, int ch_num, int stride){
    int irow = input.row(), icol = input.col(), channel = input.channel();
    int krow = kernel.row(), kcol = kernel.col(), ksquare = krow * kcol;
    int rrow = res.row(), rcol = res.col(), rsquare = rrow * rcol;
    int kstart = ksquare * ch_begin, kend = kstart + ksquare * ch_num;
    int rstart = rsquare * ch_begin, rend = rstart + rsquare * ch_num;
    int iend = input.get_data().size();

    for(int ch = 0; ch < ch_num; ch++){
      // conv_cnt: record convolution times.
      // line_cnt: record line num already convoluted.
      // sum: 一次卷积的加和.
      // c_cnt, r_cnt: to record whether jump to next line.
      int conv_cnt = 0, line_cnt = 0, sum = 0;
      int ker_i = kstart + ch * ksquare, res_i = rstart + ch * rsquare; 
      for(int c_cnt = 0, r_cnt = 0, inp_i = 0; inp_i < iend;){
        sum += kernel[ker_i++] * input[inp_i++];
        if(++c_cnt == kcol){          // cross a col
          inp_i += icol - kcol;
          c_cnt = 0;
          if(++r_cnt == krow){        // finish a conv
            // plus to res
            res[res_i++] += sum;
            ker_i = r_cnt = sum = 0;

            // handle inp_i
            conv_cnt += stride;
            inp_i = conv_cnt;
            if(conv_cnt % rcol == 0){ // switch to next line
              inp_i = ++line_cnt * stride * icol;
            }
          }
        }
      }
    }
    puts("In conv2d");
  }
}