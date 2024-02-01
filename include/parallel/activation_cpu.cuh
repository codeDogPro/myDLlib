#pragma once

#include <basic/tensor_macro.cuh>
#include <data/tensor.cuh>
#include <cmath>

namespace dl{

  template<typename T>
  bool relu_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    for(int i = start; i < end; i++){
      (*output)[i] = (*input)[i] > 0 ? (*input)[i] : 0;
    }
    return true;
  }

  template<typename T>
  bool sigmoid_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    for(int i = start; i < end; i++){
      (*output)[i] = 1 / (1 + std::exp(-(*input)[i])) + eps;
    }
    return true;
  }

  template<typename T>
  bool exp_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    for(int i = start; i < end; i++){
      (*output)[i] = std::exp((*input)[i]);
    }
    return true;
  }

  template<typename T>
  bool softmax_axis0_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int row = input->row(), col = input->col();
    std::vector<T> sums(task_num * row, T(0));
    for(int idx = start, s_idx = 0; idx < end; s_idx ++){
      for(int c_idx = 0; c_idx < col; c_idx ++){
        sums[s_idx] += (*input)[idx ++];
      }
    }
    for(int idx = start, s_idx = 0; idx < end; s_idx ++){
      for(int c_cnt = 0; c_cnt < col; c_cnt++, idx ++){
        (*output)[idx] = (*input)[idx] / sums[s_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool softmax_axis1_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int start = offset + shape * task_begin, end = start + shape * task_num;
    int row = input->row(), col = input->col();
    std::vector<T> sums(task_num * col, T(0));
    for(int idx = start; idx < end;){
      for(int c_idx = 0; c_idx < col; c_idx ++){
        sums[c_idx] += (*input)[idx ++];
      }
    }
    for(int idx = start; idx < end;){
      for(int c_idx = 0; c_idx < col; c_idx ++, idx ++){
        (*output)[idx] = (*input)[idx] / sums[c_idx];
      }
    }
    return true;
  }

  template<typename T>
  bool softmax_axis2_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input){ 
    int row = input->row(), col = input->col(), channel = input->channel();
    int start = offset + shape * task_begin;
    std::vector<T> sums(task_num * col, T(0));
    for(int ch = 0, idx = start; ch < channel; ch ++){
      for(int r_idx = 0; r_idx < task_num; r_idx++){
        int r_offset = r_idx * col;
        for(int c_idx = 0; c_idx < col; c_idx ++){
          sums[r_offset + c_idx] += (*input)[idx++];
        }
      }
      idx += (row - task_num) * col;
    }
    for(int ch = 0, idx = start; ch < channel; ch ++){
      for(int r_idx = 0; r_idx < task_num; r_idx++){
        int r_offset = r_idx * col;
        for(int c_idx = 0; c_idx < col; c_idx ++, idx ++){
          (*output)[idx] = (*input)[idx] / sums[r_offset + c_idx];
        }
      }
      idx += (row - task_num) * col;
    }
    return true;
  }
}