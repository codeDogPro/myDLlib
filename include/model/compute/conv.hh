#pragma once

#include <basic/function.hh>
#include <parallel/conv_parallel.hh>


namespace dl{

#define CONV_DEBUG

template<typename T=f32>
class Conv1D : public Function<T> {

};


template<typename T=f32>
class Conv2D : public Function<T> {
public:
  explicit 
  Conv2D (int kernel_size, int input_ch, int output_ch, int stride=1, int padding=0)
  : M_weight(kernel_size, kernel_size, input_ch, output_ch), M_bias(output_ch, 1) {
    M_stride    = stride;
    M_padding    = padding;
    M_kernelSize = kernel_size;
  #ifdef CONV_DEBUG
    std::cout << "weight:\n" << M_weight << std::endl;
    std::cout << "bias:\n" << M_bias << std::endl;
  #endif
  }

  explicit 
  Conv2D(Tensor<T>& weight, Tensor<T> &bias, int stride=1, int padding=0) {
    M_weight = std::move(weight);
    M_bias = std::move(bias);
    M_stride    = stride;
    M_padding    = padding;
    M_kernelSize = M_weight.row();
  #ifdef CONV_DEBUG
    std::cout << "weight:\n" << M_weight << std::endl;
    std::cout << "bias:\n" << M_bias << std::endl;
  #endif
  }

  virtual ~Conv2D(){};

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
    int row = input->row(), col = input->col(), channel = input->channel();
    int number = input->number();
    if(M_padding){
      auto pad_input = std::make_shared<Tensor<T>>
      (row + 2 * M_padding, col + 2 * M_padding, channel, number, 0);
      for(int i = 0; i < number; i++){
        int offset = i * number;
        parallelizer.parallel_channel(padding_parallel<T>,
         pad_input, offset, input, M_padding);
      }
      parallelizer.sync();
  #ifdef CONV_DEBUG
      std::cout << "pad_input:\n" << *pad_input << std::endl;
  #endif
      return conv_boost(pad_input, res_row(row), res_col(col));
    }

    return conv_boost(input, res_row(row), res_col(col));
  }

  int nstride() { return M_stride; }
  int npadding() { return M_padding; }

protected:
  int res_row(int row){return (row - M_weight.row() + 2 * M_padding)/M_stride + 1;}
  int res_col(int col){return (col - M_weight.col() + 2 * M_padding)/M_stride + 1;}

  std::shared_ptr<Tensor<T>> 
  conv_boost(const std::shared_ptr<Tensor<T>> input, int o_row, int o_col){
    int irow = input->row(), icol = input->col(), ichannel = input->channel();
    int number = input->number(), ivolume = irow * icol * ichannel;
    int output_ch = M_weight.number();
    // std::cout << "output_ch:" << output_ch << std::endl;
    auto output = std::make_shared<Tensor<T>>(o_row, o_col, output_ch, number, 0);
    for(int i = 0; i < number; i++){
      int offset = i * ivolume;
      if(M_kernelSize == 1){
        puts("In 1x1");
        parallelizer.parallel_channel(conv2d_1x1_parallel<T>, output, offset, input, 
          M_weight, M_bias, M_stride);
      }
      else{
        parallelizer.parallel_channel(conv2d_parallel<T>, output, offset, input, 
          M_weight, M_bias, M_stride);
      }
    }
    parallelizer.sync();
    return output;
  }


private:
  int M_padding;
  int M_stride;
  int M_kernelSize;
  Tensor<T> M_weight, M_bias;
};

}