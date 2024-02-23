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
           std::shared_ptr<const Tensor<T>> weight,
           std::shared_ptr<const Tensor<T>> bias){
  int arow = input->row(), acol = input->col(), channel = input->channel();
  int brow = acol, bcol = weight->col();
  int ooffset = ioffset/acol * bcol;
  int a_idx = ioffset + task_begin, b_idx = 0; 
  int o_idx = ooffset + task_begin;
  std::unique_lock lck(parallelizer.get_mutex(), std::defer_lock);
  if (bcol >= 8) {
    int align_bcol = bcol - bcol % 8;
    f32 *out_addr = reinterpret_cast<f32 *>(&((*output)[0]));
    const f32 *b_addr = reinterpret_cast<const f32 *>(&((*weight)[0]));
    for (int b_y = task_begin; b_y < task_begin + task_num; b_y++) {
      for (int b_x = 0; b_x < align_bcol; b_x += 8) {
        f32 *output_src = out_addr + o_idx + b_x;
        __m256 _a = _mm256_set1_ps((*input)[a_idx]);
        __m256 _b = _mm256_load_ps(b_addr + b_idx + b_x);
        __m256 mul_res = _mm256_mul_ps(_a, _b);
        /*the load output action need to lock the output data*/
        lck.lock();
        __m256 opt_raw = _mm256_load_ps(output_src);
        lck.unlock();
        __m256 accum = _mm256_add_ps(opt_raw, mul_res);
        /*the store output action need to lock the output data*/
        lck.lock();
        _mm256_storeu_ps(output_src, accum);
        lck.unlock();
      }
      // the rest of elems that can't be mod by 8
      for (int b_x = align_bcol; b_x < bcol; b_x++) {
        // shorten the Critical Zone
        T value = (*input)[a_idx] * (*weight)[b_idx + b_x];
        lck.lock();
        (*output)[o_idx + b_x] += value;
        lck.unlock();
      }
      a_idx ++, b_idx += bcol;
    }
  } else {
    // bcol < 8 no need to use simd(avx2)
    for (int b_y = task_begin; b_y < task_begin + task_num; b_y++) {
      for (int b_x = 0; b_x < bcol; b_x++) {
        // shorten the Critical Zone
        T value = (*input)[a_idx] * (*weight)[b_idx + b_x];
        lck.lock();
        (*output)[o_idx + b_x] += value;
        lck.unlock();
      }
      a_idx ++, b_idx += bcol;
    }
  }
  return true;
}

} // namespace dl