#pragma once

#include <basic/function.hh>
#include <data/tensor.hh>
#include <parallel/conv_boost.hh>


namespace dl{

template<typename T>
class Conv1D : public Function<T> {

};


template<typename T>
class Conv2D : public Function<T> {
public:
  explicit Conv2D
  (int size, int input_ch, int output_ch=1, 
   int stride=1, int paddle=0, bool auto_grad=false) {
    M_parameter = Tensor<T>(size, size, input_ch, -1, output_ch);
    M_stride    = stride;
    M_paddle    = paddle;
    M_auto_grad = auto_grad;
    //debug
    std::cout << M_parameter;
  }

  explicit 
  Conv2D(Tensor<T>& kernel, int stride=1, int paddle=0, bool auto_grad=false) {
    M_parameter = kernel;
    M_stride    = stride;
    M_paddle    = paddle;
    M_auto_grad = auto_grad;
    //debug
    std::cout << M_parameter;
  }

  virtual ~Conv2D(){};

  virtual Tensor<T>
  forward(const Tensor<T> &input){
    int row = input.row(), col = input.col(), channel = input.channel();
    if(M_paddle){
      Tensor<T> pad_input(row + 2 * M_paddle, col + 2 * M_paddle, channel, 0);
      paddle(input, pad_input, M_paddle);
      puts("In pad_conv");
      std::cout << pad_input;
      return conv_boost(pad_input, res_row(row), res_col(col));
    }
    puts("In conv");
    if(M_auto_grad) M_grad = input;

    return conv_boost(input, res_row(row), res_col(col));
  }

  int nstride() { return M_stride; }
  int npaddle() { return M_paddle; }

protected:
  int res_row(int row){return (row - M_parameter.row() + 2 * M_paddle)/M_stride + 1;}
  int res_col(int col){return (col - M_parameter.col() + 2 * M_paddle)/M_stride + 1;}

  Tensor<T> 
  conv_boost(const Tensor<T> &input, int r_row, int r_col){
    int irow = input.row(), icol = input.col(), channel = input.channel();
    int output_ch = M_parameter.number();
    std::cout << "output_ch:" << output_ch << std::endl;
    Tensor<T> res(r_row, r_col, output_ch, 0);

    int ncpu = cpu_number();
    if(output_ch >= ncpu * BOOST_CONV / 8){
    // output channel is big enough, so boost it.
      parallel_channel(conv2d_channel<T>, 
      /*nthread, res */NTHREAD_C(ncpu, output_ch), res,
      /*const args...*/input, M_parameter, M_stride);
    }
    else if(irow >= ncpu * BOOST_CONV){
    // input size is big enough, so boost it.

    }
    else{
    // no need to boost
      puts("no boost");
      conv2d_channel(0, output_ch, res, 
                     input, M_parameter, M_stride);
    }
    return res;
  }


private:
  bool M_auto_grad;
  int M_paddle;
  int M_stride;
  Tensor<T> M_parameter, M_grad;
};

}