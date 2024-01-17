#pragma once

#include <basic/tensor_macro.hh>
#include <data/tensor.hh>

#include <vector>
#include <iostream>
#include <cassert>

// for simd
#include <immintrin.h>

namespace dl{

//##################### Thread functions ########################
  template<typename T=f32>
  bool tensor_copy
  (int task_begin, int task_num, int shape, int offset,  
  std::shared_ptr<Tensor<T>> lhs, const Tensor<T> &rhs){
    int start = offset + task_begin * shape, end = start + task_num * shape;
    for(int idx = start; idx < end; idx++){
      // TODO: use simd
      (*lhs)[idx] = rhs[idx];
    }
    return true;
  }

  template<typename T>
  bool vec_add_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int arow = a.row(), col = a.col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
            __m256 res[2];
            for(int offset = 0; offset < 2; offset ++){
              __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[a_i + 8 * offset]));
              __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[b_i + 8 * offset]));
              res[offset] = _mm256_add_ps(_a, _b);
            }
            for(int offset = 0; offset < 2; offset ++){
              _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), res[offset]);
            }
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    else{ // no need to use simd
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i++, b_i++){
            (*output)[a_i] = a[a_i] + b[b_i];
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_sub_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int arow = a.row(), col = a.col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
            __m256 res[2];
            for(int offset = 0; offset < 2; offset ++){
              __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[a_i + 8 * offset]));
              __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[b_i + 8 * offset]));
              res[offset] = _mm256_sub_ps(_a, _b);
            }
            for(int offset = 0; offset < 2; offset ++){
              _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), res[offset]);
            }
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    else{ // no need to use simd
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i++, b_i++){
            (*output)[a_i] = a[a_i] - b[b_i];
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_mul_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int arow = a.row(), col = a.col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
            __m256 res[2];
            for(int offset = 0; offset < 2; offset ++){
              __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[a_i + 8 * offset]));
              __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[b_i + 8 * offset]));
              res[offset] = _mm256_mul_ps(_a, _b);
            }
            for(int offset = 0; offset < 2; offset ++){
              _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), res[offset]);
            }
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    else{ // no need to use simd
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i++, b_i++){
            (*output)[a_i] = a[a_i] * b[b_i];
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_div_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int arow = a.row(), col = a.col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
            __m256 res[2];
            for(int offset = 0; offset < 2; offset ++){
              __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[a_i + 8 * offset]));
              __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[b_i + 8 * offset]));
              res[offset] = _mm256_div_ps(_a, _b);
            }
            for(int offset = 0; offset < 2; offset ++){
              _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), res[offset]);
            }
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    else{ // no need to use simd
      for(int ch = task_begin; ch < task_begin + task_num; ch++){
        for(int a_i = astart; a_i < aend; ){
          for(int b_i = bstart; b_i < bend; a_i++, b_i++){
            (*output)[a_i] = a[a_i] / b[b_i];
          }
        }
        astart += shape, bstart += col;
        aend += shape, bend += bstart + col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_add_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    // align to use simd
    int align_end = end - end % 16;
    for(int i = start; i < align_end; i += 16){
      __m256 res[2];
      for(int offset = 0; offset < 2; offset ++){
        __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[i + 8 * offset]));
        __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[i + 8 * offset]));
        res[offset] = _mm256_add_ps(_a, _b);
      }
      for(int offset = 0; offset < 2; offset ++){
        _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])), res[offset]);
      }
    }
    // the rest of elem can't use simd
    for(int i = align_end; i < end; i++){
      (*output)[i] = a[i] + b[i];
    }
    return true;
  }

  template<typename T>
  bool vec_sub_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    // align to use simd
    int align_end = end - end % 16;
    for(int i = start; i < align_end; i += 16){
      __m256 res[2];
      for(int offset = 0; offset < 2; offset ++){
        __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[i + 8 * offset]));
        __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[i + 8 * offset]));
        res[offset] = _mm256_sub_ps(_a, _b);
      }
      for(int offset = 0; offset < 2; offset ++){
        _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])), res[offset]);
      }
    }
    // the rest of elem can't use simd
    for(int i = align_end; i < end; i++){
      (*output)[i] = a[i] - b[i];
    }
    return true;
  }

  template<typename T>
  bool vec_mul_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    // align to use simd
    int align_end = end - end % 16;
    for(int i = start; i < align_end; i += 16){
      __m256 res[2];
      for(int offset = 0; offset < 2; offset ++){
        __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[i + 8 * offset]));
        __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[i + 8 * offset]));
        res[offset] = _mm256_mul_ps(_a, _b);
      }
      for(int offset = 0; offset < 2; offset ++){
        _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])), res[offset]);
      }
    }
    // the rest of elem can't use simd
    for(int i = align_end; i < end; i++){
      (*output)[i] = a[i] * b[i];
    }
    return true;
  }

  template<typename T>
  bool vec_div_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const Tensor<T> &a, const Tensor<T> &b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    // align to use simd
    int align_end = end - end % 16;
    for(int i = start; i < align_end; i += 16){
      __m256 res[2];
      for(int offset = 0; offset < 2; offset ++){
        __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&a[i + 8 * offset]));
        __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(&b[i + 8 * offset]));
        res[offset] = _mm256_div_ps(_a, _b);
      }
      for(int offset = 0; offset < 2; offset ++){
        _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])), res[offset]);
      }
    }
    // the rest of elem can't use simd
    for(int i = align_end; i < end; i++){
      (*output)[i] = a[i] / b[i];
    }
    return true;
  }



  template<typename T>
  bool operator_sum_axis0
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(); shape *= col; 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / col + task_begin * row;
    for(int idx = start; idx < end; res_i++){
      for(int col_cnt = 0; col_cnt < col; col_cnt ++){
        (*output)[res_i] += self[idx++];
      }
    }
    return true;
  }


  template<typename T>
  bool operator_mean_axis0
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(); shape *= col;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / col + task_begin * row;
    for(int idx = start; idx < end; res_i++){
      for(int col_cnt = 0; col_cnt < col; col_cnt ++){
        (*output)[res_i] += self[idx++];
      }
      (*output)[res_i] /= col;
    }
    return true;
  }

  template<typename T>
  bool operator_max_axis0
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    shape *= col;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / col + task_begin * row;
    for(int idx = start; idx < end;){
      std::vector<T> maxs(row, INT_MIN);
      for(int row_idx = 0; row_idx < row; row_idx ++){
        for(int col_cnt = 0; col_cnt < col; col_cnt ++){
          maxs[row_idx] = std::max(maxs[row_idx], self[idx++]);
        }
        (*output)[res_i ++] = maxs[row_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_min_axis0
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    shape *= col;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / col + task_begin * row;
    for(int idx = start; idx < end;){
      std::vector<T> mins(row, INT_MAX);
      for(int row_idx = 0; row_idx < row; row_idx ++){
        for(int col_cnt = 0; col_cnt < col; col_cnt ++){
          mins[row_idx] = std::min(mins[row_idx], self[idx++]);
        }
        (*output)[res_i ++] = mins[row_idx ++];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_sum_axis1
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(); shape *= row;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / row + task_begin * col;
    for(int idx = start; idx < end;){
      std::vector<T> sums(col, 0);
      for(int row_cnt = 0; row_cnt < row; row_cnt++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          sums[col_idx] += self[idx++];
        }
      }
      for(int col_idx = 0; col_idx < col; col_idx++){
        (*output)[res_i ++] = sums[col_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_mean_axis1
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(); shape *= row;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / row + task_begin * col;
    for(int idx = start; idx < end;){
      std::vector<T> sums(col, 0);
      for(int row_cnt = 0; row_cnt < row; row_cnt++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          sums[col_idx] += self[idx++];
        }
      }
      for(int col_idx = 0; col_idx < col; col_idx++){
        (*output)[res_i ++] = sums[col_idx] / row;
      }
    }
    return true;
  }

  template<typename T>
  bool operator_max_axis1
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col();  shape *= row;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / row + task_begin * col;
    for(int idx = start; idx < end;){
      std::vector<T> maxs(col, INT_MIN);
      for(int row_cnt = 0; row_cnt < row; row_cnt++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          maxs[col_idx] = std::max(maxs[col_idx], self[idx++]);
        }
      }
      for(int col_idx = 0; col_idx < col; col_idx++){
        (*output)[res_i ++] = maxs[col_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_min_axis1
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(); shape *= row; 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int res_i = offset / row + task_begin * col;
    for(int idx = start; idx < end;){
      std::vector<T> mins(col, INT_MAX);
      for(int row_cnt = 0; row_cnt < row; row_cnt++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          mins[col_idx] = std::min(mins[col_idx], self[idx++]);
        }
      }
      for(int col_idx = 0; col_idx < col; col_idx++){
        (*output)[res_i ++] = mins[col_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_sum_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    shape *= channel;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    std::vector<T> sums(row * col, 0);
    for(int idx = start; idx < end;){
      for(int row_idx = 0; row_idx < row; row_idx++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          sums[row_idx * col + col_idx] += self[idx++];
        }
      }
    }
    int res_offset = offset / channel;
    for(int row_idx = 0; row_idx < row; row_idx++){
      for(int col_idx = 0; col_idx < col; col_idx ++){
        int idx = row_idx * col + col_idx; 
        (*output)[res_offset + idx] = sums[idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_mean_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    shape *= channel;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    std::vector<T> sums(row * col, 0);
    for(int idx = start; idx < end;){
      for(int row_idx = 0; row_idx < row; row_idx++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          sums[row_idx * col + col_idx] += self[idx++];
        }
      }
    }
    int res_offset = offset / channel;
    for(int row_idx = 0; row_idx < row; row_idx++){
      for(int col_idx = 0; col_idx < col; col_idx ++){
        int idx = row_idx * col + col_idx; 
        (*output)[res_offset + idx] = sums[idx] / channel;
      }
    }
    return true;
  }

  template<typename T>
  bool operator_max_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    shape *= channel;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    std::vector<T> maxs(row * col, INT_MIN);
    for(int idx = start; idx < end;){
      for(int row_idx = 0; row_idx < row; row_idx++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          int max_idx = row_idx * col + col_idx;
          maxs[max_idx] = std::max(maxs[max_idx], self[idx++]);
        }
      }
    }
    int res_offset = offset / channel;
    for(int row_idx = 0; row_idx < row; row_idx++){
      for(int col_idx = 0; col_idx < col; col_idx ++){
        int idx = row_idx * col + col_idx; 
        (*output)[res_offset + idx] = maxs[idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_min_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    shape *= channel;
    int start = offset + shape * task_begin, end = start + shape * task_num;
    std::vector<T> mins(row * col, INT_MAX);
    for(int idx = start; idx < end;){
      for(int row_idx = 0; row_idx < row; row_idx++){
        for(int col_idx = 0; col_idx < col; col_idx ++){
          int min_idx = row_idx * col + col_idx;
          mins[min_idx] = std::min(mins[min_idx], self[idx++]);
        }
      }
    }
    int res_offset = offset / channel;
    for(int row_idx = 0; row_idx < row; row_idx++){
      for(int col_idx = 0; col_idx < col; col_idx ++){
        int idx = row_idx * col + col_idx; 
        (*output)[res_offset + idx] = mins[idx];
      }
    }
    return true;
  }

} // namespace dl
