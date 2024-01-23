#pragma once

#include <data/tensor.hh>
#include <basic/tensor_macro.hh>

// for simd
#include <immintrin.h>

namespace dl{
  template<typename T>
  bool matMul_channel_parallel(int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output,
   const std::shared_ptr<Tensor<T>> a, const std::shared_ptr<Tensor<T>> b){
    int arow = a->row(), acol = a->col(), channel = a->channel();
    int brow = acol, bcol = b->col();
    int asquare = arow * acol, avolume = asquare * channel;
    int bsquare = brow * bcol, bvolume = bsquare * channel;
    int osquare = arow * bcol, ovolume = osquare * channel;
    int a_start = offset * avolume + task_begin * asquare; 
    int b_start = offset * bvolume + task_begin * bsquare; 
    int o_start = offset * ovolume + task_begin * osquare;
    if(bcol >= 8){
      f32 *out_address = reinterpret_cast<f32*>(&(*output)[0]);
      const f32 *b_address = reinterpret_cast<const f32*>(&(*b)[0]);
      int align_bcol = bcol - bcol % 8;
      for(int ch = task_begin; ch < task_begin + task_num; ch ++){
        int a_idx = a_start + ch * asquare; 
        int o_idx = o_start + ch * osquare;
        for(int a_y = 0; a_y < arow; a_y++) {
          int b_idx = b_start + ch * bsquare; 
          for(int b_y = 0; b_y < brow; b_y ++){
            for(int b_x = 0; b_x < align_bcol; b_x += 8){
              f32 *output_src = out_address + o_idx + b_x;
              __m256 _a = _mm256_set1_ps((*a)[a_idx]);
              __m256 _b = _mm256_load_ps(b_address + b_idx + b_x);
              __m256 mul_res = _mm256_mul_ps(_a, _b);
              __m256 out = _mm256_load_ps(output_src);
              __m256 accum = _mm256_add_ps(out, mul_res);
              _mm256_storeu_ps(output_src, accum);
            }
            // the rest of elems that can't be mod by 8
            for(int b_x = align_bcol; b_x < bcol; b_x ++){
              (*output)[o_idx + b_x] += (*a)[a_idx] * (*b)[b_idx + b_x];
            }
            a_idx ++, b_idx += bcol;
          }
          o_idx += bcol;
        }
      }
    }
    else{
      // bcol < 8 no need to use simd(avx2)
      for(int ch = task_begin; ch < task_begin + task_num; ch ++){
        int a_idx = a_start + ch * asquare; 
        int o_idx = o_start + ch * osquare;
        for(int a_y = 0; a_y < arow; a_y++) {
          int b_idx = b_start + ch * bsquare; 
          for(int b_y = 0; b_y < brow; b_y ++){
            for(int b_x = 0; b_x < bcol; b_x ++){
              (*output)[o_idx + b_x] += (*a)[a_idx] * (*b)[b_idx + b_x];
            }
            a_idx ++, b_idx += bcol;
          }
          o_idx += acol;
        }
      }
    }
    return true;
  }

  template<typename T>
  bool matMul_row_parallel(int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output,
   const std::shared_ptr<Tensor<T>> a, const std::shared_ptr<Tensor<T>> b){
    int arow = a->row(), acol = a->col(), channel = a->channel();
    int brow = acol, bcol = b->col();
    int a_idx = task_begin * acol, o_idx = task_begin * bcol;
    if(bcol >= 8){
      int align_bcol = bcol - bcol % 8;
      f32 *out_address = reinterpret_cast<f32*>(&((*output)[0]));
      const f32 *b_address = reinterpret_cast<const f32*>(&((*b)[0]));
      for(int a_y = 0; a_y < task_num; a_y++) {
        int b_idx = 0;
        for(int b_y = 0; b_y < brow; b_y ++){
          for(int b_x = 0; b_x < align_bcol; b_x += 8){
            f32 *output_src = out_address + o_idx + b_x;
            __m256 _a = _mm256_set1_ps((*a)[a_idx]);
            __m256 _b = _mm256_load_ps(b_address + b_idx + b_x);
            __m256 mul_res = _mm256_mul_ps(_a, _b);
            __m256 out = _mm256_load_ps(output_src);
            __m256 accum = _mm256_add_ps(out, mul_res);
            _mm256_storeu_ps(output_src, accum);
          }
          // the rest of elems that can't be mod by 8
          for(int b_x = align_bcol; b_x < bcol; b_x ++){
            (*output)[o_idx + b_x] += (*a)[a_idx] * (*b)[b_idx + b_x];
          }
          a_idx ++, b_idx += bcol;
        }
        o_idx += bcol;
      }
    }
    else{
      // bcol < 8 no need to use simd(avx2)
      for(int a_y = 0; a_y < task_num; a_y++) {
        int b_idx = 0; 
        for(int b_y = 0; b_y < brow; b_y ++){
          for(int b_x = 0; b_x < bcol; b_x ++){
            (*output)[o_idx + b_x] += (*a)[a_idx] * (*b)[b_idx + b_x];
          }
          a_idx ++, b_idx += bcol;
        }
        o_idx += acol;
      }
    }
    return true;
  }
}