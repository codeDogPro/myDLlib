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
    Conv2D(int size, int stride=1, int channel=1, int paddle=0, bool auto_grad=false)
    {
      m_parameter = Tensor<T>(size, size, channel);
      std::cout << m_parameter;
      m_stride = stride;
      m_paddle = paddle;
      mauto_grad = auto_grad;
    }
  
    virtual Tensor<T>
    forward(Tensor<T> &input){
      // if(mauto_grad) grad = input;
      int row = input.row(), col = input.col(), channel = input.channel();
      Tensor<T> res(res_row(row), res_col(col), m_parameter.channel(), 0);
      std::cout << res;
      if(m_paddle){
        Tensor<T> pad_input(row + 2 * m_paddle, col + 2 * m_paddle, channel, 0);
        paddle(input, pad_input, m_paddle);
        conv_boost(pad_input, res);
        puts("Finish pad_conv");
        return res;
      }
      conv_boost(input, res);
      puts("Finish conv");
      return res;
    }

    int nstride() { return m_stride; }
    int npaddle() { return m_paddle; }
  
  protected:
    int res_row(int row){return (row - m_parameter.row() + 2 * m_paddle)/m_stride + 1;}
    int res_col(int col){return (col - m_parameter.col() + 2 * m_paddle)/m_stride + 1;}

    void 
    conv_boost(Tensor<T> &input, Tensor<T> &res){
      int channel = m_parameter.channel();
      int ncpu = std::thread::hardware_concurrency();
      std::vector<std::thread> pool;
      if(channel >= ncpu * BOOST_CONV){
        int nth = NTHREAD_C(ncpu), ch_num = channel / nth , ch_mod = channel % nth;
        for(int i = 0; i < nth; i++){
          int ch_begin = ch_num * i;
          std::thread task(conv2d_channel<T>, std::ref(input), std::ref(m_parameter), 
                           std::ref(res), ch_begin, ch_num, m_stride); 
          pool.push_back(std::move(task));
        }
        if(ch_mod){
          int ch_begin = channel - ch_mod;
          conv2d_channel(input, m_parameter, res, ch_begin, ch_mod, m_stride);
        } goto join;
      }
      conv2d_channel(input, m_parameter, res, 0, res.channel(), m_stride);
      puts("In here");

    join:
      for(auto &task : pool) task.join();
    }


  private:
    bool mauto_grad;
    int m_paddle;
    int m_stride;
    Tensor<T> m_parameter, grad;
  };
}