#pragma once

#include <basic/tensor_macro.hh>

namespace dl{

  template<typename T>
  bool maxPooling_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input,
   int pool_size, int stride){
    int irow = input->row(),  icol = input->col(), channel = input->channel();
    int orow = output->row(), ocol = output->col();
    int isquare = irow * icol, osquare = orow * ocol;
    int i_start = task_begin * isquare, o_start = task_begin * osquare;

    int i_idx = i_start, o_idx = o_start;
    int x_end = icol - pool_size, y_end = irow - pool_size;
    for(int ch = task_begin; ch < task_begin + task_num; ch ++){
      i_idx = ch * isquare;
      for(int y_idx = 0; y_idx <= y_end; y_idx += stride){
        for(int x_idx = 0; x_idx <= x_end; x_idx += stride){
          for(int kr = 0; kr < pool_size; kr ++){
            int input_idx = i_idx + kr * icol;
            for(int kc = 0; kc < pool_size; kc ++){
              (*output)[o_idx] = std::max((*output)[o_idx], (*input)[input_idx + kc]);
            } // kernel col loop
          } // kernel row loop
          o_idx ++, i_idx += stride;
        } // stride x loop
        i_idx += (stride - 1) * icol;
      } // stride y loop
    } // input channel loop
    return true;
  }

  template<typename T>
  bool avgPooling_parallel
  (int task_begin, int task_num, int shape, int offset,
   std::shared_ptr<Tensor<T>> output, const std::shared_ptr<Tensor<T>> input,
   int pool_size, int stride){
    int irow = input->row(),  icol = input->col(), channel = input->channel();
    int orow = output->row(), ocol = output->col();
    int isquare = irow * icol, osquare = orow * ocol;
    int i_start = task_begin * isquare, o_start = task_begin * osquare;
    f32 avg = pool_size * pool_size;

    int i_idx = i_start, o_idx = o_start;
    int x_end = icol - pool_size, y_end = irow - pool_size;
    for(int ch = task_begin; ch < task_begin + task_num; ch ++){
      i_idx = ch * isquare;
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
        i_idx += (stride - 1) * icol;
      } // stride y loop
    } // input channel loop
    return true;
  }
}