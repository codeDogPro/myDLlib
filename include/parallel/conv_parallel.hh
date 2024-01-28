#pragma once

#include <data/tensor.cuh>
#include <basic/tensor_macro.cuh>

// for simd
#include <immintrin.h>

namespace dl{

  template<typename T>
  bool padding_parallel
  (int task_begin, int task_num, int shape, int ioffset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input, 
   int npaddle){
    int irow = input->row(), icol = input->col(), ichannel = input->channel();
    int orow = output->row(), ocol = output->col();
    int input_i = ioffset + task_begin * irow * icol;
    int ooffset = ioffset / (irow * icol) * orow * ocol;
    int output_i = ooffset + npaddle * (ocol + 1) + task_begin * orow * ocol;
    if(icol >= 8){
      if(icol % 8 == 0){
        int icol_align = icol - icol % 8;
        const f32 *input_addr = reinterpret_cast<f32 *>(&(*input)[0]);
        f32 *output_addr = reinterpret_cast<f32 *>(&(*output)[0]);
        for(int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++){
          for(int row_cnt = 0; row_cnt < irow; row_cnt++){
            for(int col_cnt = 0; col_cnt < icol_align; col_cnt += 8){
              _mm256_storeu_ps(output_addr + output_i, 
                _mm256_load_ps(input_addr + input_i));
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
      else{ // not align to 32B
        int icol_align = icol - icol % 8;
        const f32 *input_addr = reinterpret_cast<f32 *>(&(*input)[0]);
        f32 *output_addr = reinterpret_cast<f32 *>(&(*output)[0]);
        for(int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++){
          for(int row_cnt = 0; row_cnt < irow; row_cnt++){
            for(int col_cnt = 0; col_cnt < icol_align; col_cnt += 8){
              _mm256_storeu_ps(output_addr + output_i, 
                _mm256_loadu_ps(input_addr + input_i));
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
  (int task_begin, int task_num, int shape, int ioffset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input,
    const Tensor<T> &weight, const Tensor<T> &bias, int stride) {
    int irow = input->row(),  icol = input->col(), ichannel = input->channel();
    int krow = weight.row(), kcol = weight.col(), kchannel = weight.channel();
    int orow = output->row(), ocol = output->col(), ochannel = output->channel();
    int isquare = irow * icol, ksquare = krow * kcol, osquare = orow * ocol;
    int ooffset = orow * ocol * ochannel * ioffset / (irow * icol * ichannel);
    // xx_start：表示内层循环的起始，需要与偏移相加得到最终的索引xxx_idx
    int o_start = ooffset + task_begin * osquare;
    int k_start = task_begin * krow * kcol * kchannel;

    // bias add
    for(int ch = task_begin; ch < task_begin + task_num; ch++){
      for(int idx = 0; idx < osquare; idx++){
        (*output)[o_start + idx] = bias[ch];
      }
      o_start += osquare;
    }

    // weight conv
    int x_end = icol - kcol, y_end = irow - krow;
    for(int n = task_begin; n < task_begin + task_num; n++){
      for(int ch_i = 0; ch_i < ichannel; ch_i ++){
        int i_idx = ioffset + ch_i * isquare, o_idx = ooffset + n * osquare;
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

  template<typename T=f32>
  bool conv2d_1x1_parallel
  (int task_begin, int task_num, int shape, int ioffset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input,
    const Tensor<T> &weight, const Tensor<T> &bias, int stride) {
    int row = input->row(), col = input->col();
    int ichannel = input->channel(), ochannel = output->channel();
    int square = row * col, kchannel = weight.channel();
    int ooffset = ochannel * ioffset / ichannel;
    // xx_start：表示内层循环的起始，需要与偏移相加得到最终的索引xxx_idx
    int o_start = ooffset + task_begin * square;
    int k_start = task_begin * kchannel;

    // bias add
    for(int ch = task_begin; ch < task_begin + task_num; ch++){
      for(int idx = 0; idx < square; idx++){
        (*output)[o_start + idx] = bias[ch];
      }
      o_start += square;
    }

    // weight conv
    for(int n = task_begin; n < task_begin + task_num; n++){
      for(int ch_i = 0; ch_i < ichannel; ch_i ++){
        int i_idx = ioffset + ch_i * square, o_idx = ooffset + n * square;
        int w_idx = n * kchannel + ch_i;
        for(int y_idx = 0; y_idx < row; y_idx += stride){
          for(int x_idx = 0; x_idx < col; x_idx += stride){
            (*output)[o_idx] += (*input)[i_idx] * weight[w_idx];
            o_idx += stride, i_idx += stride;
          } // stride x loop
          i_idx += (stride - 1) * col;
        } // stride y loop
      } // input channel loop
    } // kernel number loop
    return true;
  }

}
