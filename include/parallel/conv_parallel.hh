#pragma once

#include <data/tensor.hh>
#include <basic/tensor_macro.hh>

// for simd
#include <immintrin.h>

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
    if(icol >= 8){
      int icol_align = icol - icol % 8;
      for(int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++){
        for(int row_cnt = 0; row_cnt < irow; row_cnt++){
          for(int col_cnt = 0; col_cnt < icol_align; col_cnt += 8){
            _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[output_i])),
              _mm256_loadu_ps(reinterpret_cast<const f32 *>(&(*input)[input_i])));
            input_i += 8, output_i += 8;
          }
          for(int col_cnt = icol_align; col_cnt < icol; col_cnt ++){
            (*output)[output_i++] = (*input)[input_i++];
          }
          output_i += 2 * npaddle;
        }
        output_i += 2 * npaddle * ocol;
      }
    } 
    else{
      for(int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++){
        for(int row_cnt = 0; row_cnt < irow; row_cnt++){
          for(int col_cnt = 0; col_cnt < icol; col_cnt++){
            (*output)[output_i++] = (*input)[input_i++];
          }
          output_i += 2 * npaddle;
        }
        output_i += 2 * npaddle * ocol;
      }
    }
    return true;
  }

  template<typename T=f32>
  bool conv2d_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input,
    const Tensor<T> &weight, const Tensor<T> &bias, int stride) {
    int irow = input->row(),  icol = input->col(), ichannel = input->channel();
    int krow = weight.row(), kcol = weight.col(), kchannel = weight.channel();
    int orow = output->row(), ocol = output->col(), ochannel = output->channel();
    int isquare = irow * icol, ksquare = krow * kcol, osquare = orow * ocol;
    // xx_start：表示内层循环的起始，需要与偏移相加得到最终的索引xxx_idx
    int o_start = task_begin * osquare;
    int k_start = task_begin * krow * kcol * kchannel;

    // bias add
    for(int ch = task_begin; ch < task_begin + task_num; ch++){
      for(int idx = 0; idx < osquare; idx++){
        (*output)[o_start + idx] = bias[ch];
      }
      o_start += osquare;
    }

    // weight conv
    int o_idx = task_begin * osquare;
    int x_end = icol - kcol, y_end = irow - krow;
    for(int n = task_begin; n < task_begin + task_num; n++){
      int i_idx = 0; 
      for(int ch_i = 0; ch_i < ichannel; ch_i ++){
        i_idx = ch_i * isquare, o_idx = n * osquare;
        for(int y_idx = 0; y_idx <= y_end; y_idx += stride){
          for(int x_idx = 0; x_idx <= x_end; x_idx += stride){
            for(int kr = 0; kr < krow; kr ++){
              int input_idx = i_idx + kr * icol;
              int weight_idx = k_start + kr * kcol;
              for(int kc = 0; kc < kcol; kc ++){
                (*output)[o_idx] += (*input)[input_idx + kc] * weight[weight_idx + kc];
              } // kernel col loop
            } // kernel row loop
            o_idx += stride, i_idx += stride;
          } // stride x loop
          i_idx += (stride - 1) * icol + kcol - 1;
        } // stride y loop
        k_start += ksquare;
      } // input channel loop
    } // kernel number loop

    return true;
  }
}
