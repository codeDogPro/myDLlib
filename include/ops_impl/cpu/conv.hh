#pragma once

#include <basic/tensor_macro.cuh>
#include <data/tensor.cuh>

#include <ops_impl/cpu/avx_lib.hh>

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
  const int64_t ooffset = ioffset/(irow*icol*ich) * (orow*ocol*och);
  // xx_start：表示内层循环的起始，需要与偏移相加得到最终的索引xxx_idx
  int64_t o_start = ooffset + task_begin * osquare;
  int k_start = task_begin * krow * kcol * kch;

  // bias add
  for (int ch = task_begin; ch < task_begin + task_num; ch++) {
    for (int idx = 0; idx < osquare; idx++) {
      //TODO: use simd
      (*output)[o_start + idx] = bias[ch];
    }
    o_start += osquare;
  }

  // weight conv
  const int x_end = icol - kcol, y_end = irow - krow;
  for (int n = task_begin; n < task_begin + task_num; n++) {
    for (int ch_i = 0; ch_i < ich; ch_i++) {
      int64_t i_idx = ioffset + ch_i * isquare, o_idx = ooffset + n * osquare;
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
bool conv2d_k1_cpu(int task_begin, int task_num, int shape, int ioffset,
                    std::shared_ptr<Tensor<T>> output,
                    const std::shared_ptr<const Tensor<T>> input,
                    const Tensor<T> &weight, const Tensor<T> &bias,
                    int stride) {
  const int irow = input->row(), icol = input->col();
  const int orow = output->row(), ocol = output->col();
  const int ichannel = input->channel(), ochannel = output->channel();
  const int isquare = irow * icol, osquare = orow * ocol;
  const int kchannel = weight.channel();
  const int64_t ooffset = (ioffset/ichannel) * ochannel;
  // xx_start：表示内层循环的起始，需要与偏移相加得到最终的索引xxx_idx
  int64_t o_start = ooffset + task_begin * osquare;

  // bias add
  for (int ch_o = task_begin; ch_o < task_begin + task_num; ch_o++) {
    for (int idx = 0; idx < osquare; idx++) {
      (*output)[o_start + idx] = bias[ch_o];
    }
    o_start += osquare;
  }

  // weight conv
  for (int ch_o = task_begin; ch_o < task_begin + task_num; ch_o++) {
    for (int ch_i = 0; ch_i < ichannel; ch_i++) {
      int64_t i_idx = ioffset + ch_i * isquare;
      int64_t o_idx = ooffset + ch_o * osquare;
      T reg_w = weight[ch_o*kchannel + ch_i]; // store weight in register
      for (int y_idx = 0; y_idx < irow; y_idx += stride) {
        for (int x_idx = 0; x_idx < icol; x_idx += stride) {
          (*output)[o_idx] += (*input)[i_idx] * reg_w;
          o_idx++, i_idx += stride;
        } // stride x loop
        i_idx += (stride - 1) * icol;
      } // stride y loop
    }   // input channel loop
  }     // kernel number loop
  return true;
}

//* kernel_size=1 stride=1 implementaion
template <typename T = f32>
bool conv2d_k1s1_cpu(int task_begin, int task_num, int shape, int ioffset,
                    std::shared_ptr<Tensor<T>> output,
                    const std::shared_ptr<const Tensor<T>> input,
                    const Tensor<T> &weight, const Tensor<T> &bias) {
  const int row = input->row(), col = input->col();
  const int ichannel = input->channel(), ochannel = output->channel();
  const int square = row * col, kchannel = weight.channel();

  const int64_t ooffset = (ioffset/ichannel) * ochannel;
  int64_t o_start = ooffset + task_begin * square;

  // bias add
  for (int ch_o = task_begin; ch_o < task_begin + task_num; ch_o++) {
    // __m256 reg_b = _mm256_set1_ps(bias[ch_o]); 
    for (int idx = 0; idx < square; idx++) {
      (*output)[o_start + idx] = bias[ch_o];
    }
    o_start += square;
  }

  //*weight conv, parallel base on output channel
  if(square >= 16){//* Use avx2 */
    f32 *out_addr = reinterpret_cast<f32 *>(&(*output)[0]);
    const f32 *inp_addr = reinterpret_cast<const f32 *>(&(*input)[0]);
    //* kernel number loop(output channel loop)
    for (int ch_o = task_begin; ch_o < task_begin+task_num; ch_o++) {
      //* input channel loop
      for (int ch_i = 0; ch_i < ichannel; ch_i++) {
        const int64_t i_idx = ioffset + ch_i * square;
        int64_t o_idx = ooffset + ch_o * square;
        const int align_beg = 8 - i_idx%8;
        const int align_end = square - (square-align_beg)%16;
        __m256 reg256_w = _mm256_set1_ps(weight[ch_o*kchannel + ch_i]); 
        //* unaligned front part, because of elems < 8, 
        //* so just use a unaligned simd to finish these elems.
        if(align_beg != 0){
          __m256 inp = _mm256_loadu_ps(inp_addr + i_idx);
          __m256 mul = _mm256_mul_ps(inp, reg256_w);
          __m256 out = _mm256_loadu_ps(out_addr + o_idx);
          __m256 res = _mm256_add_ps(out, mul);
          _mm256_storeu_ps(out_addr + o_idx, res);
        }
        // o_idx += align_beg;
        //* aligned part that can use simd
        for(int i = align_beg; i < align_end; i += 16){
          for(int ofst = 0; ofst < 2; ofst ++){
            __m256 inp = _mm256_load_ps(inp_addr + (i_idx+i+ofst*8));
            __m256 mul = _mm256_mul_ps(inp, reg256_w);
            // TODO: align bug o_idx=196 row=col=14
            __m256 out = _mm256_load_ps(out_addr + (o_idx+i+ofst*8));
            __m256 res = _mm256_add_ps(out, mul);
            _mm256_store_ps(out_addr + (o_idx+i+ofst*8), res);
          }
          // o_idx += 16;
        }
        //* unaligned end part 
        for(int i = align_end; i < square; i++){
          (*output)[o_idx + i] = (*input)[i_idx + i] * reg256_w[0];
        }
      } 
    } 
  } else{   //* square < 16, so no need to use avx2
    //TODO: not implemented
  }
  return true;
}

} // namespace dl
