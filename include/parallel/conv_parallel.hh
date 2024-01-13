#pragma once

#include <data/tensor.hh>
#include <basic/tensor_macro.hh>

namespace dl{

  template<typename T>
  bool paddle_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input, 
   int npaddle){
    int irow = input->row(), icol = input->col(), ichannel = input->channel();
    int orow = output->row(), ocol = output->col();
    int input_i = offset + task_begin * irow * icol;
    int output_i = offset + npaddle * (ocol + 1) + task_begin * orow * ocol;
    for(int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++){
      for(int row_cnt = 0; row_cnt < irow; row_cnt++){
        for(int col_cnt = 0; col_cnt < icol; col_cnt++){
          (*output)[output_i++] = (*input)[input_i++];
        }
        output_i += 2 * npaddle;
      }
      output_i += 2 * npaddle * ocol;
    }
    return true;
  }

  template<typename T=f32>
  bool conv2d_parallel
  (int n_begin, int n_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input,
    const Tensor<T> &weight, const Tensor<T> &bias, int stride) {
    int irow = input->row(),  icol = input->col(),  isquare = irow * icol;
    int krow = weight.row(), kcol = weight.col(), ksquare = krow * kcol;
    int orow = output->row(), ocol = output->col(), osquare = orow * ocol;
    int kvolume = ksquare * weight.channel();
    int kstart = kvolume * n_begin, kend = kstart + kvolume * n_num;
    int rstart = osquare * n_begin, rend = rstart + osquare * n_num;
    int iend = input->size();
    // printf("output use_count: %ld\n", output.use_count());

    //TODO:need to change
    for(int n = 0; n < n_num; n++){
      // conv_cnt: record convolution times.
      // line_cnt: record line num that already convoluted.
      // ch_cnt:   record channel num that already convoluted.
      // sum: 一次卷积的加和.
      // c_cnt, r_cnt: to record whether jump to next line.
      int conv_cnt = 0, line_cnt = 0, ch_cnt = 0;
      int ker_offset = kvolume * n, inp_offset = 0, ch_offset = 0;
      int ker_i = kstart + ker_offset, res_i = rstart + osquare * n; 
      for(T sum = 0, c_cnt = 0, r_cnt = 0, inp_i = 0; inp_i < iend;){
        // std::cout << inp_i << ' ';
        sum += (*input)[inp_i++] * weight[ker_i++];
        if(++c_cnt == kcol){          // cross a col
          inp_i += icol - kcol;
          c_cnt = 0;
          if(++r_cnt == krow){        // finish a conv
            // plus to res
            // std::cout << "res_i:" << res_i << ' ';
            (*output)[res_i++] += sum;
            // std::cout << '\n' << (*output)[res_i - 1];
            r_cnt = sum = 0;
            
            // handle inp_i and ker_i
            conv_cnt += stride;
            inp_i = inp_offset + conv_cnt;
            ker_i = ker_offset + ch_cnt * ksquare;
            if((conv_cnt / stride) % ocol == 0){ // switch to next line
              // printf("Switch to next line. conv_cnt=%d\n\n", conv_cnt);
              inp_i = inp_offset = ch_cnt * ch_offset + (++line_cnt * stride * icol);
              conv_cnt = 0;
              if(line_cnt == orow){          //  switch to next input channel
                // printf("Switch to next channel. line_cnt=%d\n\n", line_cnt);
                inp_i = inp_offset = ch_offset = ++ch_cnt * isquare;
                ker_i = ker_offset + ch_cnt * ksquare;
                res_i = osquare * n;
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