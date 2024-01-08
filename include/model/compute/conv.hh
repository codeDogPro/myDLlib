#pragma once

#include <basic/function.hh>
#include <parallel/conv_parallel.hh>


namespace dl{

template<typename T>
class Conv1D : public Function<T> {

};


template<typename T=f32>
class Conv2D : public Function<T> {
public:
  explicit 
  Conv2D (int kernel_size, int input_ch, int output_ch=1, int stride=1, int paddle=0) {
    M_weight = Tensor<T>(kernel_size, kernel_size, input_ch, -1, output_ch);
    M_bias = Tensor<T>(output_ch, 1);
    M_stride    = stride;
    M_paddle    = paddle;
    //debug
    std::cout << "weight:\n" << M_weight << std::endl;
    std::cout << "bias:\n" << M_bias << std::endl;
  }

  explicit 
  Conv2D(Tensor<T>& weight, Tensor<T> &bias, int stride=1, int paddle=0) {
    M_weight = std::move(weight);
    M_bias = std::move(bias);
    M_stride    = stride;
    M_paddle    = paddle;
    //debug
    std::cout << "weight:\n" << M_weight << std::endl;
    std::cout << "bias:\n" << M_bias << std::endl;
  }

  virtual ~Conv2D(){};

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
    int row = input->row(), col = input->col(), channel = input->channel();
    if(M_paddle){
      auto pad_input = std::make_shared<Tensor<T>>
      (row + 2 * M_paddle, col + 2 * M_paddle, channel, 0);
      parallelizer.parallel_channel(paddle_parallel<T>, pad_input, input, M_paddle);
      // std::cout << "pad_input:\n" << *pad_input << std::endl;
      return conv_boost(pad_input, res_row(row), res_col(col));
    }

    return conv_boost(input, res_row(row), res_col(col));
  }

  int nstride() { return M_stride; }
  int npaddle() { return M_paddle; }

protected:
  int res_row(int row){return (row - M_weight.row() + 2 * M_paddle)/M_stride + 1;}
  int res_col(int col){return (col - M_weight.col() + 2 * M_paddle)/M_stride + 1;}

  std::shared_ptr<Tensor<T>> 
  conv_boost(const std::shared_ptr<Tensor<T>> input, int r_row, int r_col){
    int irow = input->row(), icol = input->col(), channel = input->channel();
    int output_ch = M_weight.number();
    // std::cout << "output_ch:" << output_ch << std::endl;
    auto output = std::make_shared<Tensor<T>>(r_row, r_col, output_ch, 0);
    parallelizer.parallel_channel(conv2d_parallel<T>, 
                                  output, input, 
                                  M_weight, M_bias, M_stride);
    return output;
  }


private:
  int M_paddle;
  int M_stride;
  Tensor<T> M_weight, M_bias;
};

}