#pragma once

#include <basic/tensor_macro.cuh>
#include <data/tensor.cuh>

// for simd
#include <immintrin.h>
// for opencv
#include <opencv2/core.hpp>

namespace dl{

  template<typename T=f32>
  bool tensor_copy
  (int task_begin, int task_num, int shape, int offset,  
  std::shared_ptr<Tensor<T>> lhs, const Tensor<T> &rhs){
    int start = offset + task_begin * shape, end = start + task_num * shape;
    if(rhs.col() >= 16){
      int align_start = start + 16 - (start % 16);
      int align_end = end - end % 16;
      // the start of elem can't use simd
      for(int i = start; i < align_start; i++){
        (*lhs)[i] = rhs[i];
      }
      // aligned idx, so it can use _mm256_load_ps
      for(int i = align_start; i < align_end; i += 16){
        __m256 tmp[2];
        for(int offset = 0; offset < 2; offset ++){
          tmp[offset] = _mm256_load_ps(reinterpret_cast<const f32 *>(&rhs[i + 8 * offset]));
        }
        for(int offset = 0; offset < 2; offset ++){
          _mm256_store_ps(reinterpret_cast<f32 *>(&(*lhs)[i + 8 * offset]), tmp[offset]);
        }
      }
      // the rest of elem can't use simd
      for(int i = align_end; i < end; i++){
        (*lhs)[i] = rhs[i];
      }
    }
    else{
      for(int idx = start; idx < end; idx++){
        (*lhs)[idx] = rhs[idx];
      }
    }
    return true;
  }

  template<typename T=f32>
  bool cvMat2Tensor
  (int task_begin, int task_num, int shape, int offset,  
  std::shared_ptr<Tensor<T>> output, const cv::Mat &mat){
    int col = output->col(), row = output->row(), channel = output->channel();
    int square = row * col, depth = mat.depth();
    int start = offset + task_begin * shape, end = start + task_num * shape;
    int o_idx = start;
    switch(depth){
      case CV_8U:
      case CV_8S: // 没有专门的Vec3xxx对应，索性和uchar一起吧
        if(channel == 3){
          auto it_start = mat.begin<cv::Vec3b>() + start;
          auto it_end = mat.begin<cv::Vec3b>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            for(int ch = 0; ch < channel; ch ++){
              (*output)[ch * square + o_idx] = static_cast<T>((*it)[ch]);
            }
          } 
        }
        else if(channel == 1){
          auto it_start = mat.begin<uchar>() + start;
          auto it_end = mat.begin<uchar>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            (*output)[o_idx] = static_cast<T>(*it);
          } 
        } break;
      case CV_16U:
        if(channel == 3){
          auto it_start = mat.begin<cv::Vec3w>() + start;
          auto it_end = mat.begin<cv::Vec3w>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            for(int ch = 0; ch < channel; ch ++){
              (*output)[ch * square + o_idx] = static_cast<T>((*it)[ch]);
            }
          }
        }
        else if(channel == 1){
          auto it_start = mat.begin<ushort>() + start;
          auto it_end = mat.begin<ushort>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            (*output)[o_idx] = static_cast<T>(*it);
          }
        } break;
      case CV_32S:
        if(channel == 3){
          auto it_start = mat.begin<cv::Vec3i>() + start;
          auto it_end = mat.begin<cv::Vec3i>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            for(int ch = 0; ch < channel; ch ++){
              (*output)[ch * square + o_idx] = static_cast<T>((*it)[ch]);
            }
          } 
        }
        else if(channel == 1){
          auto it_start = mat.begin<int>() + start;
          auto it_end = mat.begin<int>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            (*output)[o_idx] = static_cast<T>(*it);
          } 
        } break;
      case CV_32F:
        if(channel == 3){
          auto it_start = mat.begin<cv::Vec3f>() + start;
          auto it_end = mat.begin<cv::Vec3f>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            for(int ch = 0; ch < channel; ch ++){
              (*output)[ch * square + o_idx] = static_cast<T>((*it)[ch]);
            }
          } 
        }
        else if(channel == 1){
          auto it_start = mat.begin<f32>() + start;
          auto it_end = mat.begin<f32>() + end;
          for(auto it = it_start; it != it_end; it++, o_idx ++){
            (*output)[o_idx] = static_cast<T>(*it);
          } 
        } break;
      default: exit(-1);
    }
    return true;
  }

  template<typename T>
  bool vec_add_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int arow = output->row(), col = output->col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      if(col % 16 == 0){
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_add_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
      }
      else{ // not align to 64B
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          int align_bend = bend - col % 16;
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < align_bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_loadu_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_loadu_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_add_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
            // finish the rest of elements that can't use simd
            for(int b_i = align_bend; b_i < bend; a_i++, b_i++){
              (*output)[a_i] = a[a_i] + b[b_i];
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
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
        aend += shape,   bend += col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_sub_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int arow = output->row(), col = output->col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      if(col % 16 == 0){
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_sub_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
      }
      else{ // not align to 64B
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          int align_bend = bend - col % 16;
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < align_bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_sub_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
            // finish the rest of elements that can't use simd
            for(int b_i = align_bend; b_i < bend; a_i++, b_i++){
              (*output)[a_i] = a[a_i] - b[b_i];
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
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
        aend += shape,   bend += col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_mul_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int arow = output->row(), col = output->col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      if(col % 16 == 0){
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_mul_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
      }
      else{ // not align to 64B
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          int align_bend = bend - col % 16;
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < align_bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_mul_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
            // finish the rest of elements that can't use simd
            for(int b_i = align_bend; b_i < bend; a_i++, b_i++){
              (*output)[a_i] = a[a_i] * b[b_i];
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
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
        aend += shape,   bend += col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_div_single
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int arow = output->row(), col = output->col();
    int astart = offset + task_begin * shape, aend = astart + shape;
    int bstart = offset / arow + task_begin * col, bend = bstart + col; 
    if(col >= 16){
      if(col % 16 == 0){
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_div_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
      }
      else{ // not align to 64B
        for(int ch = task_begin; ch < task_begin + task_num; ch++){
          int align_bend = bend - col % 16;
          for(int a_i = astart; a_i < aend; ){
            for(int b_i = bstart; b_i < align_bend; a_i += 16, b_i += 16){
              __m256 res[2];
              for(int offset = 0; offset < 2; offset ++){
                __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+a_i+8*offset));
                __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+b_i+8*offset));
                res[offset] = _mm256_div_ps(_a, _b);
              }
              for(int offset = 0; offset < 2; offset ++){
                _mm256_storeu_ps(reinterpret_cast<f32 *>(&((*output)[a_i + 8 * offset])), 
                  res[offset]);
              }
            }
            // finish the rest of elements that can't use simd
            for(int b_i = align_bend; b_i < bend; a_i++, b_i++){
              (*output)[a_i] = a[a_i] / b[b_i];
            }
          }
          astart += shape, bstart += col;
          aend += shape,   bend += col;
        }
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
        aend += shape,   bend += col;
      }
    }
    return true;
  }

  template<typename T>
  bool vec_add_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    if(output->col() >= 16){
      int align_start = start + 16 - (start % 16);
      int align_end = end - end % 16;
      // the start of elem can't use simd
      for(int i = start; i < align_start; i++){
        (*output)[i] = a[i] + b[i];
      }
      // aligned idx, so it can use _mm256_load_ps
      for(int i = align_start; i < align_end; i += 16){
        __m256 res[2];
        for(int offset = 0; offset < 2; offset ++){
          __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+ i + 8 * offset));
          __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+ i + 8 * offset));
          res[offset] = _mm256_add_ps(_a, _b);
        }
        for(int offset = 0; offset < 2; offset ++){
          _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])), 
            res[offset]);
        }
      }
      // the rest of elem can't use simd
      for(int i = align_end; i < end; i++){
        (*output)[i] = a[i] + b[i];
      }
    }
    else{
      // tensor is too small to use simd
      for(int i = start; i < end; i++){
        (*output)[i] = a[i] + b[i];
      }
    }
    return true;
  }

  template<typename T>
  bool vec_sub_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    if(output->col() >= 16){
      int align_start = start + 16 - (start % 16);
      int align_end = end - end % 16;
      // the start of elem can't use simd
      for(int i = start; i < align_start; i++){
        (*output)[i] = a[i] - b[i];
      }
      // aligned idx, so it can use _mm256_load_ps
      for(int i = align_start; i < align_end; i += 16){
        __m256 res[2];
        for(int offset = 0; offset < 2; offset ++){
          __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+ i + 8 * offset));
          __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+ i + 8 * offset));
          res[offset] = _mm256_sub_ps(_a, _b);
        }
        for(int offset = 0; offset < 2; offset ++){
          _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])),
            res[offset]);
        }
      }
      // the rest of elem can't use simd
      for(int i = align_end; i < end; i++){
        (*output)[i] = a[i] - b[i];
      }
    }
    else{
      // tensor is too small to use simd
      for(int i = start; i < end; i++){
        (*output)[i] = a[i] - b[i];
      }
    }
    return true;
  }

  template<typename T>
  bool vec_mul_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    if(output->col() >= 16){
      int align_start = start + 16 - (start % 16);
      int align_end = end - end % 16;
      // the start of elem can't use simd
      for(int i = start; i < align_start; i++){
        (*output)[i] = a[i] * b[i];
      }
      // aligned idx, so it can use _mm256_load_ps
      for(int i = align_start; i < align_end; i += 16){
        __m256 res[2];
        for(int offset = 0; offset < 2; offset ++){
          __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+ i + 8 * offset));
          __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+ i + 8 * offset));
          res[offset] = _mm256_mul_ps(_a, _b);
        }
        for(int offset = 0; offset < 2; offset ++){
          _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])), 
            res[offset]);
        }
      }
      // the rest of elem can't use simd
      for(int i = align_end; i < end; i++){
        (*output)[i] = a[i] * b[i];
      }
    }
    else{
      // tensor is too small to use simd
      for(int i = start; i < end; i++){
        (*output)[i] = a[i] * b[i];
      }
    }
    return true;
  }

  template<typename T>
  bool vec_div_full
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const T *a, const T *b) {
    int start = offset + task_begin * shape, end = start + task_num * shape;
    if(output->col() >= 16){
      int align_start = start + 16 - (start % 16);
      int align_end = end - end % 16;
      // the start of elem can't use simd
      for(int i = start; i < align_start; i++){
        (*output)[i] = a[i] / b[i];
      }
      // aligned idx, so it can use _mm256_load_ps
      for(int i = align_start; i < align_end; i += 16){
        __m256 res[2];
        for(int offset = 0; offset < 2; offset ++){
          __m256 _a = _mm256_load_ps(reinterpret_cast<const f32 *>(a+ i + 8 * offset));
          __m256 _b = _mm256_load_ps(reinterpret_cast<const f32 *>(b+ i + 8 * offset));
          res[offset] = _mm256_div_ps(_a, _b);
        }
        for(int offset = 0; offset < 2; offset ++){
          _mm256_store_ps(reinterpret_cast<f32 *>(&((*output)[i + 8 * offset])),
            res[offset]);
        }
      }
      // the rest of elem can't use simd
      for(int i = align_end; i < end; i++){
        (*output)[i] = a[i] / b[i];
      }
    }
    else{
      // tensor is too small to use simd
      for(int i = start; i < end; i++){
        (*output)[i] = a[i] / b[i];
      }
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
      for(int r_idx = 0; r_idx < row; r_idx ++){
        for(int col_cnt = 0; col_cnt < col; col_cnt ++){
          maxs[r_idx] = std::max(maxs[r_idx], self[idx++]);
        }
        (*output)[res_i ++] = maxs[r_idx];
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
      for(int r_idx = 0; r_idx < row; r_idx ++){
        for(int col_cnt = 0; col_cnt < col; col_cnt ++){
          mins[r_idx] = std::min(mins[r_idx], self[idx++]);
        }
        (*output)[res_i ++] = mins[r_idx ++];
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
        for(int c_idx = 0; c_idx < col; c_idx ++){
          sums[c_idx] += self[idx++];
        }
      }
      for(int c_idx = 0; c_idx < col; c_idx++){
        (*output)[res_i ++] = sums[c_idx];
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
        for(int c_idx = 0; c_idx < col; c_idx ++){
          sums[c_idx] += self[idx++];
        }
      }
      for(int c_idx = 0; c_idx < col; c_idx++){
        (*output)[res_i ++] = sums[c_idx] / row;
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
        for(int c_idx = 0; c_idx < col; c_idx ++){
          maxs[c_idx] = std::max(maxs[c_idx], self[idx++]);
        }
      }
      for(int c_idx = 0; c_idx < col; c_idx++){
        (*output)[res_i ++] = maxs[c_idx];
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
        for(int c_idx = 0; c_idx < col; c_idx ++){
          mins[c_idx] = std::min(mins[c_idx], self[idx++]);
        }
      }
      for(int c_idx = 0; c_idx < col; c_idx++){
        (*output)[res_i ++] = mins[c_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_sum_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    int start = offset + shape * task_begin;
    std::vector<T> sums(task_num * col, 0);
    for(int ch = 0, idx = start; ch < channel; ch++){
      for(int r_idx = 0; r_idx < task_num; r_idx++){
        int r_offset = r_idx * col;
        for(int c_idx = 0; c_idx < col; c_idx ++){
          sums[r_offset + c_idx] += self[idx++];
        }
      }
      idx += (row - task_num) * col;
    }
    int res_offset = offset / channel;
    for(int r_idx = 0; r_idx < task_num; r_idx++){
      int o_idx = res_offset + (task_begin + r_idx) * col;
      int s_idx = r_idx * col; 
      for(int c_idx = 0; c_idx < col; c_idx ++){
        (*output)[o_idx + c_idx] = sums[s_idx + c_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_mean_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    int start = offset + shape * task_begin;
    std::vector<T> sums(task_num * col, 0);
    for(int ch = 0, idx = start; ch < channel; ch++){
      for(int r_idx = 0; r_idx < task_num; r_idx++){
        int r_offset = r_idx * col;
        for(int c_idx = 0; c_idx < col; c_idx ++){
          sums[r_offset + c_idx] += self[idx++];
        }
      }
      idx += (row - task_num) * col;
    }
    int res_offset = offset / channel;
    for(int r_idx = 0; r_idx < task_num; r_idx++){
      int o_idx = res_offset + (task_begin + r_idx) * col;
      int s_idx = r_idx * col; 
      for(int c_idx = 0; c_idx < col; c_idx ++){
        (*output)[o_idx + c_idx] = sums[s_idx + c_idx] / channel;
      }
    }
    return true;
  }

  template<typename T>
  bool operator_max_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    int start = offset + shape * task_begin;
    std::vector<T> maxs(task_num * col, INT_MIN);
    for(int ch = 0, idx = start; ch < channel; ch++){
      for(int r_idx = 0; r_idx < task_num; r_idx++){
        int r_offset = r_idx * col;
        for(int c_idx = 0; c_idx < col; c_idx ++){
          maxs[r_offset + c_idx] = std::max(maxs[r_offset + c_idx], self[idx++]);
        }
      }
      idx += (row - task_num) * col;
    }
    int res_offset = offset / channel;
    for(int r_idx = 0; r_idx < task_num; r_idx++){
      int o_idx = res_offset + (task_begin + r_idx) * col;
      int m_idx = r_idx * col; 
      for(int c_idx = 0; c_idx < col; c_idx ++){
        (*output)[o_idx + c_idx] = maxs[m_idx + c_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool operator_min_axis2
  (int task_begin, int task_num, int shape, int offset, 
  std::shared_ptr<Tensor<T>> output, const Tensor<T> &self) {
    int row = self.row(), col = self.col(), channel = self.channel(); 
    int start = offset + shape * task_begin;
    std::vector<T> mins(task_num * col, INT_MAX);
    for(int ch = 0, idx = start; ch < channel; ch++){
      for(int r_idx = 0; r_idx < task_num; r_idx++){
        int r_offset = r_idx * col;
        for(int c_idx = 0; c_idx < col; c_idx ++){
          mins[r_offset + c_idx] = std::min(mins[r_offset + c_idx], self[idx++]);
        }
      }
      idx += (row - task_num) * col;
    }
    int res_offset = offset / channel;
    for(int r_idx = 0; r_idx < task_num; r_idx++){
      int o_idx = res_offset + (task_begin + r_idx) * col;
      int m_idx = r_idx * col; 
      for(int c_idx = 0; c_idx < col; c_idx ++){
        (*output)[o_idx + c_idx] = mins[m_idx + c_idx];
      }
    }
    return true;
  }

} // namespace dl
