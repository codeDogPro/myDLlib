#pragma once

#include <data/tensor.cuh>
#include <basic/tensor_macro.cuh>

namespace dl{

/*
  matrix a: input
  matrix b: weight
*/
template<typename T>
bool 
Linear_cpu(int task_begin, int task_num, int shape, int ioffset,
           std::shared_ptr<Tensor<T>> output,
           std::shared_ptr<const Tensor<T>> input,
           const Tensor<T> &weight,
           const Tensor<T> &bias){
  const int arow = input->row(), col = input->col(), channel = input->channel();
  const int brow = weight.row();
  const int ooffset = ioffset/col * brow;
  int b_idx = task_begin * col; 
  int o_idx = ooffset + task_begin;
  int bias_idx = task_begin;
  if (col >= 8) {
    f32 *out_addr = reinterpret_cast<f32 *>(&(*output)[0]);
    const f32 *a_addr = reinterpret_cast<const f32 *>(&(*input)[ioffset]);
    const f32 *b_addr = reinterpret_cast<const f32 *>(&weight[0]);
    const int align_col = col - col % 8;
    for (int b_y = task_begin; b_y < task_begin+task_num; b_y++) {
      T sum = 0;
      for (int b_x = 0; b_x < align_col; b_x += 8) {
        __m256 _a = _mm256_load_ps(a_addr + b_x);
        __m256 _b = _mm256_loadu_ps(b_addr + b_idx + b_x); // b is impossible to be aligned
        __m256 mul_res = _mm256_mul_ps(_a, _b);
        sum += hsum256_ps_avx(mul_res);
      }
      // the rest of elems that can't be mod by 8
      for (int b_x = align_col; b_x < col; b_x++) {
        sum += (*input)[ioffset + b_x] * weight[b_idx + b_x];
      }
      // store the col sum to output
      (*output)[o_idx++] = sum + bias[bias_idx++];
      b_idx += col;
    }
  } else {
    // bcol < 8 no need to use simd(avx2)
    for (int b_y = task_begin; b_y < task_begin+task_num; b_y++) {
      for (int b_x = 0; b_x < col; b_x ++) {
        (*output)[o_idx] += (*input)[ioffset + b_x] * weight[b_idx + b_x];;
      }
      (*output)[o_idx++] += bias[bias_idx++];
      b_idx += col;
    }
  }
  return true;
}

} // namespace dl