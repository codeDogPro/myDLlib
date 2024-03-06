#pragma once

#include <basic/tensor_macro.cuh>
#include <data/tensor.cuh>

// for simd
#include <immintrin.h>

namespace dl{

template <typename T>
bool padding_cpu(int task_begin, int task_num, int shape, int ioffset,
                 std::shared_ptr<Tensor<T>> output,
                 const std::shared_ptr<const Tensor<T>> input, int npad) {
  const int irow = input->row(), icol = input->col();
  const int orow = output->row(), ocol = output->col();
  const int ooffset = ioffset/(irow*icol) * orow*ocol;
  int input_i = ioffset + task_begin * irow * icol;
  int output_i = ooffset + npad * (ocol + 1) + task_begin * orow*ocol;
  if (icol >= 8) {
    if (icol % 8 == 0) {
      int icol_align = icol - icol % 8;
      auto input_addr = reinterpret_cast<const f32 *>(&(*input)[0]);
      f32 *output_addr = reinterpret_cast<f32 *>(&(*output)[0]);
      for (int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++) {
        for (int row_cnt = 0; row_cnt < irow; row_cnt++) {
          for (int col_cnt = 0; col_cnt < icol_align; col_cnt += 8) {
            _mm256_storeu_ps(output_addr + output_i,
                             _mm256_load_ps(input_addr + input_i));
            input_i += 8, output_i += 8;
          }
          for (int col_cnt = icol_align; col_cnt < icol; col_cnt++) {
            (*output)[output_i++] = (*input)[input_i++];
          }
          output_i += 2 * npad;
        }
        output_i += 2 * npad * ocol;
      }
    } else { // not align to 32B
      int icol_align = icol - icol % 8;
      auto input_addr = reinterpret_cast<const f32 *>(&(*input)[0]);
      f32 *output_addr = reinterpret_cast<f32 *>(&(*output)[0]);
      for (int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++) {
        for (int row_cnt = 0; row_cnt < irow; row_cnt++) {
          for (int col_cnt = 0; col_cnt < icol_align; col_cnt += 8) {
            _mm256_storeu_ps(output_addr + output_i,
                             _mm256_loadu_ps(input_addr + input_i));
            input_i += 8, output_i += 8;
          }
          for (int col_cnt = icol_align; col_cnt < icol; col_cnt++) {
            (*output)[output_i++] = (*input)[input_i++];
          }
          output_i += 2 * npad;
        }
        output_i += 2 * npad * ocol;
      }
    }
  } else {
    for (int ch_idx = task_begin; ch_idx < task_begin + task_num; ch_idx++) {
      for (int row_cnt = 0; row_cnt < irow; row_cnt++) {
        for (int col_cnt = 0; col_cnt < icol; col_cnt++) {
          (*output)[output_i++] = (*input)[input_i++];
        }
        output_i += 2 * npad;
      }
      output_i += 2 * npad * ocol;
    }
  }
  return true;
}
}