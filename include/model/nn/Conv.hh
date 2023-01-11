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
    (int size, int channel, int output_ch=1, 
     int stride=1, int paddle=0, bool auto_grad=false) {
      printf("ouput_ch:%d\n", output_ch);
      m_parameter = Tensor<T>(size, size, channel, output_ch);
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
        Tensor<T> pad_input(row + 2 * m_paddle, col + 2 * m_paddle, channel, 1, 0);
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
      Tensor<T> res(r_row, r_col, output_ch, 1, 0);

      int ncpu = std::thread::hardware_concurrency();
      std::vector<std::thread> pool;
      if(output_ch >= ncpu * BOOST_CONV){
        int nth = NTHREAD_C(ncpu), ch_num = output_ch / nth , ch_mod = output_ch % nth;
        for(int i = 0; i < nth; i++){
          int ch_begin = ch_num * i;
          std::thread task(conv2d_channel<T>, std::cref(input), std::cref(m_parameter), 
                           std::ref(res), ch_begin, ch_num, m_stride); 
          pool.push_back(std::move(task));
        }
        if(ch_mod){
          int ch_begin = channel - ch_mod;
          conv2d_channel(input, m_parameter, res, ch_begin, ch_mod, m_stride);
        } goto join;
      }
      else if()
      puts("no boost");
      conv2d_channel(input, m_parameter, res, 0, channel, m_stride);

    join:
      for(auto &task : pool) task.join();

      return res;
    }


  private:
    bool mauto_grad;
    int m_paddle;
    int m_stride;
    Tensor<T> m_parameter, grad;
  };
}