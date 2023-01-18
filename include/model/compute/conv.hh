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
      m_parameter = Tensor<T>(size, size, input_ch, -1, output_ch);
      m_stride = stride;
      m_paddle = paddle;
      mauto_grad = auto_grad;
      //debug
      std::cout << m_parameter;
    }

    explicit 
    Conv2D(Tensor<T> &kernel, int stride=1, int paddle=0, bool auto_grad=false) {
      m_parameter = kernel;
      m_stride = stride;
      m_paddle = paddle;
      mauto_grad = auto_grad;
      //debug
      std::cout << m_parameter;
    }

    virtual ~Conv2D(){};
  
    virtual Tensor<T>
    forward(const Tensor<T> &input){
      if(mauto_grad) grad = input;
      int row = input.row(), col = input.col(), channel = input.channel();
      if(m_paddle){
        Tensor<T> pad_input(row + 2 * m_paddle, col + 2 * m_paddle, channel, 0);
        paddle(input, pad_input, m_paddle);
        puts("In pad_conv");
        std::cout << pad_input;
        return conv_boost(pad_input, res_row(row), res_col(col));
      }
      puts("In conv");
      return conv_boost(input, res_row(row), res_col(col));
    }

    int nstride() { return m_stride; }
    int npaddle() { return m_paddle; }
  
  protected:
    int res_row(int row){return (row - m_parameter.row() + 2 * m_paddle)/m_stride + 1;}
    int res_col(int col){return (col - m_parameter.col() + 2 * m_paddle)/m_stride + 1;}

    Tensor<T> 
    conv_boost(const Tensor<T> &input, int r_row, int r_col){
      int irow = input.row(), icol = input.col(), channel = input.channel();
      int output_ch = m_parameter.number();
      std::cout << "output_ch:" << output_ch << std::endl;
      Tensor<T> res(r_row, r_col, output_ch, 0);

      int ncpu = std::thread::hardware_concurrency();
      if(output_ch >= ncpu * BOOST_CONV / 8){
      // output channel is big enough, so boost it.
        parallel_channel(conv2d_channel<T>, 
        /*nthread, res */NTHREAD_C(ncpu, output_ch), res,
        /*const args...*/input, m_parameter, m_stride);
      }
      else if(irow >= ncpu * BOOST_CONV){
      // input size is big enough, so boost it.

      }
      else{
      // no need to boost
        puts("no boost");
        conv2d_channel(0, output_ch, res, 
                       input, m_parameter, m_stride);
      }
      return res;
    }


  private:
    bool mauto_grad;
    int m_paddle;
    int m_stride;
    Tensor<T> m_parameter, grad;
  };
}