#pragma once

#include <data/tensor.cuh>
#include <basic/tensor_macro.cuh>

// for tiled for loop
#include <parallel/morton.cuh>
// for simd
#include <ops_impl/cpu/avx_lib.hh>

namespace dl{
  
/*
  This function parallel for row, and adapt multichannel/number.
  The parameter: offset is number of channel in the 4D matrix
*/
template <typename T>
bool 
matMul4D_cpu(int task_begin, int task_num, int shape, int offset,
                   std::shared_ptr<Tensor<T>> output,
                   const std::shared_ptr<const Tensor<T>> a,
                   const std::shared_ptr<const Tensor<T>> b) {
  const int arow = a->row(), acol = a->col(), channel = a->channel();
  const int brow = acol, bcol = b->col();
  const int asquare = arow * acol, bsquare = brow * bcol;
  const int osquare = arow * bcol;
  int a_idx = offset*asquare + task_begin*acol;
  int o_idx = offset*osquare + task_begin*bcol;
  if (bcol >= 8) {
    f32 *out_address = reinterpret_cast<f32 *>(&(*output)[0]);
    const f32 *b_address = reinterpret_cast<const f32 *>(&(*b)[0]);
    int align_bcol = bcol - bcol % 8;
    for (int a_y = 0; a_y < task_num; a_y++) {
      int b_idx = offset * bsquare;
      for (int b_y = 0; b_y < brow; b_y++) {
        for (int b_x = 0; b_x < align_bcol; b_x += 8) {
          f32 *output_src = out_address + o_idx + b_x;
          __m256 _a = _mm256_set1_ps((*a)[a_idx]);
          //TODO: try to align so that we can use load_ps
          __m256 _b = _mm256_loadu_ps(b_address + b_idx + b_x);
          __m256 mul_res = _mm256_mul_ps(_a, _b);
          __m256 out = _mm256_loadu_ps(output_src);
          __m256 accum = _mm256_add_ps(out, mul_res);
          _mm256_storeu_ps(output_src, accum);
        }
        // the rest of elems that can't be mod by 8
        for (int b_x = align_bcol; b_x < bcol; b_x++) {
          (*output)[o_idx + b_x] += (*a)[a_idx] * (*b)[b_idx + b_x];
        }
        a_idx++, b_idx += bcol;
      }
      o_idx += bcol;
    }
  } else {
    // bcol < 8 no need to use simd(avx2)
    for (int a_y = 0; a_y < task_num; a_y++) {
      int b_idx = offset * bsquare;
      for (int b_y = 0; b_y < brow; b_y++) {
        for (int b_x = 0; b_x < bcol; b_x++) {
          (*output)[o_idx + b_x] += (*a)[a_idx] * (*b)[b_idx + b_x];
        }
        a_idx++, b_idx += bcol;
      }
      o_idx += acol;
    }
  }
  return true;
}


// channel base
template <typename T>
bool 
matTranspose_cpu(int task_begin, int task_num, int shape, int offset,
                 std::shared_ptr<Tensor<T>> output,
                 const std::shared_ptr<const Tensor<T>> input) {
  int row = input->row(), col = input->col(), channel = input->channel();
  if (row >= BLOCK_SIZE) {
    i32 *output_addr = reinterpret_cast<i32 *>(&(*output)[0]);
    int align_row = row - row % BLOCK_SIZE;
    int align_col = col - col % BLOCK_SIZE;
    for (int ch = task_begin; ch < task_begin + task_num; ch++) {
      int idx_base = offset + ch * shape;
      int mortonEnd = (align_row / BLOCK_SIZE) * (align_col / BLOCK_SIZE);
      for (int mortonCode = 0; mortonCode < mortonEnd; mortonCode++) {
        auto [xBase, yBase] = morton2d::decode(mortonCode);
        xBase *= BLOCK_SIZE;
        yBase *= BLOCK_SIZE;
        for (int y = yBase; y < yBase + BLOCK_SIZE; y++) {
          for (int x = xBase; x < xBase + BLOCK_SIZE; x++) {
            int o_idx = idx_base + y * col + x;
            int i_idx = idx_base + x * col + y;
            _mm_stream_si32(output_addr + o_idx,
                            static_cast<int>((*input)[i_idx]));
          }
        }
      }
    }
  } else {
    // matrix's size is too small to tiled
    for (int ch = task_begin; ch < task_begin + task_num; ch++) {
      int idx_base = offset + ch * shape;
      for (int y = 0; y < row; y++) {
        int o_idx = idx_base + y * col;
        for (int x = 0; x < col; x++) {
          int i_idx = idx_base + x * col + y;
          (*output)[o_idx + x] = (*input)[i_idx];
        }
      }
    }
  }
  return true;
}
}