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
    explicit
    Conv2D(int size, int stride=1, int channel=1, int paddle=0, bool auto_grad=false) {
      m_parameter = Tensor<T>(size, size, channel, 1);
      m_stride = stride;
      m_paddle = paddle;
      mauto_grad = auto_grad;
      //debug
      std::cout << m_parameter;
    }

    explicit
    Conv2D(Tensor<T> &kernel, int stride=1, int channel=1, int paddle=0, bool auto_grad=false) {
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
      // if(mauto_grad) grad = input;
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
      int irow = input.row(), icol = input.col(), channel = m_parameter.channel();
      printf("%d %d\n", r_row, r_col);
      Tensor<T> res(r_row, r_col, channel, 0);
      res.shape();

      int ncpu = std::thread::hardware_concurrency();
      std::vector<std::thread> pool;
      if(channel >= ncpu * BOOST_CONV){
        int nth = NTHREAD_C(ncpu), ch_num = channel / nth , ch_mod = channel % nth;
        for(int i = 0; i < nth; i++){
          int ch_begin = ch_num * i;
          std::thread task(conv2d_channel<T>, std::cref(input), std::cref(m_parameter), 
                           std::ref(res), ch_begin, ch_num, m_stride); 
          pool.push_back(std::move(task));
        }
        if(ch_mod){
          puts("mod");
          int ch_begin = channel - ch_mod;
          conv2d_channel(input, m_parameter, res, ch_begin, ch_mod, m_stride);
        } goto join;
      }
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