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
    Conv2D(int size, int stride=1, int channel=1,
           int paddle=0, bool auto_grad=false)
    {
      m_parameter = Tensor<T>(size, size, channel);
      m_stride = stride;
      m_paddle = paddle;
      mauto_grad = auto_grad;
    }
  
    virtual Tensor<T>
    forward(Tensor<T> &input){
      if(mauto_grad) grad = input;
      int row = input.row(), col = input.col(), channel = input.channel();
      Tensor<T> res(res_row(row), res_col(col), m_parameter.channel(), 1);
      if(m_paddle){
        Tensor<T> pad_input(row + 2 * m_paddle, col + 2 * m_paddle, channel, 0);
        paddle(input, pad_input, m_paddle);
        return res;
      }
      return res;
    }

    int res_row(int row){return (row - m_parameter.row() + 2 * m_paddle)/m_stride + 1;}
    int res_col(int col){return (col - m_parameter.col() + 2 * m_paddle)/m_stride + 1;}

    int nstride() { return m_stride; }
    int npaddle() { return m_paddle; }

  private:
    bool mauto_grad;
    int m_paddle;
    int m_stride;
    Tensor<T> m_parameter, grad;
  };
}