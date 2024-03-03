#pragma once

#include <basic/tensor_macro.cuh>
#include <data/tensor.cuh>

namespace dl{

template <typename T = f32>
bool conv2d_cpu(int task_begin, int task_num, int shape, int ioffset,
                std::shared_ptr<Tensor<T>> output,
                const std::shared_ptr<const Tensor<T>> input,
                const Tensor<T> &weight, const Tensor<T> &bias, int stride) {
  const int irow = input->row(), icol = input->col(), ich = input->channel();
  const int krow = weight.row(), kcol = weight.col(), kch = weight.channel();
  const int orow = output->row(), ocol = output->col(), och = output->channel();
  const int isquare = irow*icol, ksquare = krow*kcol, osquare = orow*ocol;
  const uint64_t ooffset = ioffset/(irow*icol*ich) * (orow*ocol*och);
  // xx_start：表示内层循环的起始，需要与偏移相加得到最终的索引xxx_idx
  uint64_t o_start = ooffset + task_begin * osquare;
  int k_start = task_begin * krow * kcol * kch;

  // bias add
  for (int ch = task_begin; ch < task_begin + task_num; ch++) {
    for (int idx = 0; idx < osquare; idx++) {
      (*output)[o_start + idx] = bias[ch];
    }
    o_start += osquare;
  }

  // weight conv
  const int x_end = icol - kcol, y_end = irow - krow;
  for (int n = task_begin; n < task_begin + task_num; n++) {
    for (int ch_i = 0; ch_i < ich; ch_i++) {
      uint64_t i_idx = ioffset + ch_i * isquare, o_idx = ooffset + n * osquare;
      for (int y_idx = 0; y_idx <= y_end; y_idx += stride) {
        for (int x_idx = 0; x_idx <= x_end; x_idx += stride) {
          for (int kr = 0; kr < krow; kr++) {
            int input_idx = i_idx + kr * icol;
            int weight_idx = k_start + kr * kcol;
            for (int kc = 0; kc < kcol; kc++) {
              (*output)[o_idx] +=
                  (*input)[input_idx + kc] * weight[weight_idx + kc];
            } // kernel col loop
          }   // kernel row loop
          o_idx++, i_idx += stride;
        } // stride x loop
        i_idx += (stride - 1) * icol + kcol - 1;
      } // stride y loop
      k_start += ksquare;
    } // input channel loop
  }   // kernel number loop
  return true;
}

template <typename T = f32>
bool conv2d_1x1_cpu(int task_begin, int task_num, int shape, int ioffset,
                    std::shared_ptr<Tensor<T>> output,
                    const std::shared_ptr<const Tensor<T>> input,
                    const Tensor<T> &weight, const Tensor<T> &bias,
                    int stride) {
  const int row = input->row(), col = input->col();
  const int ichannel = input->channel(), ochannel = output->channel();
  const int square = row * col, kchannel = weight.channel();
  const uint64_t ooffset = (ioffset/ichannel) * ochannel;
  // xx_start：表示内层循环的起始，需要与偏移相加得到最终的索引xxx_idx
  uint64_t o_start = ooffset + task_begin * square;

  // bias add
  for (int ch = task_begin; ch < task_begin + task_num; ch++) {
    for (int idx = 0; idx < square; idx++) {
      (*output)[o_start + idx] = bias[ch];
    }
    o_start += square;
  }

  // weight conv
  for (int n = task_begin; n < task_begin + task_num; n++) {
    for (int ch_i = 0; ch_i < ichannel; ch_i++) {
      uint64_t i_idx = ioffset + ch_i * square, o_idx = ooffset + n * square;
      int w_idx = n * kchannel + ch_i;
      for (int y_idx = 0; y_idx < row; y_idx += stride) {
        for (int x_idx = 0; x_idx < col; x_idx += stride) {
          (*output)[o_idx] += (*input)[i_idx] * weight[w_idx];
          o_idx++, i_idx += stride;
        } // stride x loop
        i_idx += (stride - 1) * col;
      } // stride y loop
    }   // input channel loop
  }     // kernel number loop
  return true;
}
}
