#pragma once

#include <basic/tensor_macro.cuh>
#include <data/tensor.cuh>

namespace dl{

template <typename T>
bool maxPool2D_cpu(int task_begin, int task_num, int shape, int ioffset,
                   std::shared_ptr<Tensor<T>> output,
                   const std::shared_ptr<const Tensor<T>> input,
                   int pool_size, int stride) {
  int irow = input->row(), icol = input->col(), channel = input->channel();
  int orow = output->row(), ocol = output->col();
  int isquare = irow * icol, osquare = orow * ocol;
  // when irow and icol can't be div by pool_size, need the vars
  int align_irow = irow - irow % stride, align_icol = icol - icol % stride;
  int ooffset = ioffset / isquare * osquare;

  int x_end = align_icol - pool_size, y_end = align_irow - pool_size;
  for (int ch = task_begin; ch < task_begin + task_num; ch++) {
    int i_idx = ioffset + ch * isquare, o_idx = ooffset + ch * osquare;
    for (int y_idx = 0; y_idx <= y_end; y_idx += stride) {
      for (int x_idx = 0; x_idx <= x_end; x_idx += stride) {
        T value = static_cast<T>(-1e8);
        for (int kr = 0; kr < pool_size; kr++) {
          int input_idx = i_idx + kr * icol;
          for (int kc = 0; kc < pool_size; kc++) {
            value = std::max(value, (*input)[input_idx + kc]);
          } // kernel col loop
        } // kernel row loop
        (*output)[o_idx++] = value; // store value
        i_idx += stride;
      } // stride x loop
      i_idx += (stride - 1) * icol + icol % stride;
    } // stride y loop
  }   // input channel loop
  return true;
}

  template <typename T>
  bool avgPool2D_cpu(int task_begin, int task_num, int shape, int ioffset,
                     std::shared_ptr<Tensor<T>> output,
                     const std::shared_ptr<Tensor<T>> input, int pool_size,
                     int stride) {
    int irow = input->row(),  icol = input->col(), channel = input->channel();
    int orow = output->row(), ocol = output->col();
    int isquare = irow * icol, osquare = orow * ocol;
    // when irow and icol can't be div by pool_size, need the vars
    int align_irow = irow - irow % stride, align_icol = icol - icol % stride; 
    int ooffset = ioffset / isquare * osquare;
    f32 avg = pool_size * pool_size;

    int x_end = align_icol - pool_size, y_end = align_irow - pool_size;
    for(int ch = task_begin; ch < task_begin + task_num; ch ++){
      int i_idx = ioffset + ch * isquare, o_idx = ooffset + ch * osquare;
      for(int y_idx = 0; y_idx <= y_end; y_idx += stride){
        for(int x_idx = 0; x_idx <= x_end; x_idx += stride){
          for(int kr = 0; kr < pool_size; kr ++){
            int input_idx = i_idx + kr * icol;
            for(int kc = 0; kc < pool_size; kc ++){
              (*output)[o_idx] += (*input)[input_idx + kc];
            } // kernel col loop
          } // kernel row loop
          (*output)[o_idx] /= avg;
          o_idx ++, i_idx += stride;
        } // stride x loop
        i_idx += (stride - 1) * icol + icol % stride;
      } // stride y loop
    } // input channel loop
    return true;
  }

  template <typename T>
  bool globalAvgPool2D_cpu(int task_begin, int task_num, int shape, int ioffset,
                           std::shared_ptr<Tensor<T>> output,
                           const std::shared_ptr<const Tensor<T>> input) {
    int square = input->row() * input->col();
    int ooffset = ioffset / square;
    for(int t = task_begin; t < task_begin + task_num; t++){
      int ibase = ioffset + t * square;
      int oidx = ooffset + t;
      for(int i = 0; i < square; i++){
        (*output)[oidx] += (*input)[ibase + i];
      }
      (*output)[oidx] /= square;
    }
    return true;
  }
}