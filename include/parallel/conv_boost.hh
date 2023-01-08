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
  void conv2d_channel
  (const Tensor<T> &input, const Tensor<T> &kernel, Tensor<T> &res,
   int ch_begin, int ch_num, int stride)
  {
    int irow = input.row(),  icol = input.col(),  isquare = irow * icol;
    int krow = kernel.row(), kcol = kernel.col(), ksquare = krow * kcol;
    int rrow = res.row(),    rcol = res.col(),    rsquare = rrow * rcol;
    int kstart = ksquare * ch_begin, kend = kstart + ksquare * ch_num;
    int rstart = rsquare * ch_begin, rend = rstart + rsquare * ch_num;
    int iend = input.size();
    printf("stride:%d isquare:%d\n", stride, isquare);

    for(int ch = 0; ch < ch_num; ch++){
      // conv_cnt: record convolution times.
      // line_cnt: record line num already convoluted.
      // ch_cnt: record channel num already convoluted.
      // sum: 一次卷积的加和.
      // c_cnt, r_cnt: to record whether jump to next line.
      int conv_cnt = 0, line_cnt = 0, ch_cnt = 0, inp_offset = 0, ch_offset = 0;
      int ker_i = kstart + ch * ksquare, res_i = rstart + ch * rsquare; 
      for(int sum = 0, c_cnt = 0, r_cnt = 0, inp_i = 0; inp_i < iend;){
        std::cout << inp_i << ' ';
        sum += kernel[ker_i++] * input[inp_i++];
        if(++c_cnt == kcol){          // cross a col
          inp_i += icol - kcol;
          c_cnt = 0;
          if(++r_cnt == krow){        // finish a conv
            // plus to res
            // std::cout << "res_i:" << res_i << ' ';
            res[res_i++] += sum;
            std::cout << '\n' << res;
            // res.shape();
            ker_i = r_cnt = sum = 0;

            // handle inp_i
            conv_cnt += stride;
            inp_i = inp_offset + conv_cnt;
            if((conv_cnt / stride) % rcol == 0){ // switch to next line
              printf("Switch to next line. conv_cnt=%d\n\n", conv_cnt);
              inp_i = inp_offset = ch_cnt * ch_offset + ++line_cnt * stride * icol;
              conv_cnt = 0;
              if(line_cnt == rrow){          //  switch to next input channel
                printf("Switch to next channel. line_cnt=%d\n\n", line_cnt);
                inp_i = inp_offset = ch_offset = ++ch_cnt * isquare;
                res_i = ch * rsquare;
                line_cnt = 0;
              }
            }
          }
        }
      }
    }
  }
}