#pragma once

namespace dl{

  template<typename T> class Tensor;

  template<typename T>
  void
  paddle(const Tensor<T> &input, Tensor<T> &pad_input, int npaddle){
    int irow = input.row(), icol = input.col(), ichannel = input.channel();
    int prow = pad_input.row(), pcol = pad_input.col();
    int pad_i = npaddle * (pcol + 1);
    for(int c_cnt = 0, r_cnt = 0; auto x : input.get_cdata()){
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
  int conv2d_channel
  (int n_begin, int n_num, Tensor<T> &res,
   const Tensor<T> &input, const Tensor<T> &kernel, int stride) {
    int irow = input.row(),  icol = input.col(),  isquare = irow * icol;
    int krow = kernel.row(), kcol = kernel.col(), ksquare = krow * kcol;
    int rrow = res.row(),    rcol = res.col(),    rsquare = rrow * rcol;
    int kvolume = ksquare * kernel.channel();
    int kstart = kvolume * n_begin, kend = kstart + kvolume * n_num;
    int rstart = rsquare * n_begin, rend = rstart + rsquare * n_num;
    int iend = input.size();

    for(int n = 0; n < n_num; n++){
      // conv_cnt: record convolution times.
      // line_cnt: record line num that already convoluted.
      // ch_cnt:   record channel num that already convoluted.
      // sum: 一次卷积的加和.
      // c_cnt, r_cnt: to record whether jump to next line.
      int conv_cnt = 0, line_cnt = 0, ch_cnt = 0;
      int ker_offset = kvolume * n, inp_offset = 0, ch_offset = 0;
      int ker_i = kstart + ker_offset, res_i = rstart + rsquare * n; 
      for(int sum = 0, c_cnt = 0, r_cnt = 0, inp_i = 0; inp_i < iend;){
        // std::cout << inp_i << ' ';
        sum += kernel[ker_i++] * input[inp_i++];
        if(++c_cnt == kcol){          // cross a col
          inp_i += icol - kcol;
          c_cnt = 0;
          if(++r_cnt == krow){        // finish a conv
            // plus to res
            // std::cout << "res_i:" << res_i << ' ';
            res[res_i++] += sum;
            // std::cout << '\n' << res;
            r_cnt = sum = 0;

            // handle inp_i and ker_i
            conv_cnt += stride;
            inp_i = inp_offset + conv_cnt;
            ker_i = ker_offset + ch_cnt * ksquare;
            if((conv_cnt / stride) % rcol == 0){ // switch to next line
              // printf("Switch to next line. conv_cnt=%d\n\n", conv_cnt);
              inp_i = inp_offset = ch_cnt * ch_offset + (++line_cnt * stride * icol);
              conv_cnt = 0;
              if(line_cnt == rrow){          //  switch to next input channel
                // printf("Switch to next channel. line_cnt=%d\n\n", line_cnt);
                inp_i = inp_offset = ch_offset = ++ch_cnt * isquare;
                ker_i = ker_offset + ch_cnt * ksquare;
                res_i = rsquare * n;
                line_cnt = 0;
              }
            }
          }// krow if
        } // kcol if
      }  // for inside loop
    }   // for n_num loop
    return n_begin;
  }
}