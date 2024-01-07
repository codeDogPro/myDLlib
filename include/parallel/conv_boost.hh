#pragma once

#include <data/tensor.hh>
#include <basic/tensor_macro.hh>

namespace dl{

  template<typename T>
  void paddle(const std::shared_ptr<Tensor<T>> input,
  std::shared_ptr<Tensor<T>> pad_input, int npaddle){
    int irow = input->row(), icol = input->col(), ichannel = input->channel();
    int prow = pad_input->row(), pcol = pad_input->col();
    int pad_i = npaddle * (pcol + 1);
    auto pad_input_data = pad_input->get_data();
    //TODO:need to change
    for(int c_cnt = 0, r_cnt = 0; const auto& x : input->get_cdata()){
      pad_input_data[pad_i++] = x;
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

  template<typename T=f32>
  bool conv2d_channel
  (int n_begin, int n_num, std::shared_ptr<Tensor<T>> output,
   const std::shared_ptr<Tensor<T>> input, const Tensor<T> &weight,
   const Tensor<T> &bias, int stride) {
    size_t irow = input->row(),  icol = input->col(),  isquare = irow * icol;
    size_t krow = weight.row(), kcol = weight.col(), ksquare = krow * kcol;
    size_t rrow = output->row(), rcol = output->col(), rsquare = rrow * rcol;
    size_t kvolume = ksquare * weight.channel();
    size_t kstart = kvolume * n_begin, kend = kstart + kvolume * n_num;
    size_t rstart = rsquare * n_begin, rend = rstart + rsquare * n_num;
    size_t iend = input->size();
    auto input_data = input->get_cdata();
    auto output_data = input->get_data();

    //TODO:need to change
    for(int n = 0; n < n_num; n++){
      // conv_cnt: record convolution times.
      // line_cnt: record line num that already convoluted.
      // ch_cnt:   record channel num that already convoluted.
      // sum: 一次卷积的加和.
      // c_cnt, r_cnt: to record whether jump to next line.
      int conv_cnt = 0, line_cnt = 0, ch_cnt = 0;
      int ker_offset = kvolume * n, inp_offset = 0, ch_offset = 0;
      int ker_i = kstart + ker_offset, res_i = rstart + rsquare * n; 
      for(T sum = 0, c_cnt = 0, r_cnt = 0, inp_i = 0; inp_i < iend;){
        // std::cout << inp_i << ' ';
        sum += weight[ker_i++] * input_data[inp_i++];
        if(++c_cnt == kcol){          // cross a col
          inp_i += icol - kcol;
          c_cnt = 0;
          if(++r_cnt == krow){        // finish a conv
            // plus to res
            // std::cout << "res_i:" << res_i << ' ';
            output_data[res_i++] += sum;
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
    return true;
  }
}