#pragma once

#include <data/tensor.hh>
#include <parallel/matrix_parallel.hh>

namespace dl{
  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  matMul(const std::shared_ptr<Tensor<T>> a, const std::shared_ptr<Tensor<T>> b){
    int arow = a->row(), acol = a->col(), achannel = a->channel();
    int brow = b->row(), bcol = b->col(), bchannel = b->channel();
    int anumber = a->number(), bnumber = b->number();
    int asquare = arow * acol, avolume = asquare * achannel;
    int bsquare = brow * bcol, bvolume = bsquare * bchannel;
    assert(achannel == bchannel && anumber == bnumber);

    if(acol == brow){
      auto output = std::make_shared<Tensor<T>>(arow, bcol, achannel, anumber, 0);
      if(anumber == 1 && achannel == 1){
        parallelizer.parallel_row(matMul_row_parallel<T>, output,
          0, a, b);
      }
      else{
        for(int i = 0; i < anumber; i++){
          parallelizer.parallel_channel(matMul_channel_parallel<T>, output,
            i, a, b);
        }
      }
      parallelizer.sync();
      return output;
    }
    else{
      fprintf(stderr, "mat a's col != mat b's row!\n");
      exit(-1);
    }
  }
}