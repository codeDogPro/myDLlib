#pragma once

// #define BENCH

#include <data/rand_init.hh>
#include <parallel/parallel.hh>
#ifdef BENCH
#include <basic/timer.hh>
#endif

#include <vector>
#include <numeric>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <assert.h>

namespace dl{

template<typename T>
class Tensor{

public:
  explicit Tensor() = default;

  explicit
  Tensor(int row, int col, int channel=1, int number=1, T val=-1){
    assert(row != 0 && col != 0 && channel != 0 && number != 0);

    m_data.assign(row * col * channel * number, val);
    m_shape.assign({row, col, channel, number});
    if(val == -1){ rand_init(*this);}
  }

  explicit
  Tensor(const std::vector<int> &shape, T val=-1){
    assert(shape.size() != 0);

    int product = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    m_data.assign(product, val);
    m_shape = shape;
    if(val == -1){ rand_init(*this);}
  }

  explicit
  Tensor(std::vector<int> &data, std::vector<int> &shape)
  : m_data(data), m_shape(shape){}

  // deep copy
  explicit
  Tensor(const Tensor<T> &t){ 
    m_shape.assign(3, 0); m_data.assign(t.size(), 0);
    for(int i = 0; int x : t.get_cshape()) m_shape[i++] = x;
    for(int i = 0; int x : t.get_cdata()) m_data[i++] = x;
  }

  // move copy ctor
  Tensor(Tensor<T> &&t){ 
    m_data  = t.get_data();
    m_shape = t.get_shape();
  }

  Tensor<T> &
  operator=(const Tensor<T> &t){
    m_shape.assign(4, 0); m_data.assign(t.size(), 0);
    for(int i = 0; int x : t.get_cshape()) m_shape[i++] = x;
    for(int i = 0; int x : t.get_cdata()) m_data[i++] = x;
    return *this;
  }

  Tensor<T> &
  operator=(Tensor<T> &&t){
    m_data  = std::move(t.get_data());
    m_shape = std::move(t.get_shape());
    return *this;
  }

  Tensor<T> 
  operator+(T x){ Tensor<T> t(m_shape, x); return tensor_calculator(*this, t, PLUS);}
  Tensor<T> 
  operator-(T x){ Tensor<T> t(m_shape, x); return tensor_calculator(*this, t, MINUS);}
  Tensor<T> 
  operator*(T x){ Tensor<T> t(m_shape, x); return tensor_calculator(*this, t, MULTIPLY);}
  Tensor<T> 
  operator/(T x){ Tensor<T> t(m_shape, x); return tensor_calculator(*this, t, DIVIDE);}
  Tensor<T> 
  operator%(T x){ Tensor<T> t(m_shape, x); return tensor_calculator(*this, t, MOD);}
  Tensor<T> operator+(const Tensor<T> &t){ return tensor_calculator(*this, t, PLUS);}
  Tensor<T> operator-(const Tensor<T> &t){ return tensor_calculator(*this, t, MINUS);}
  Tensor<T> operator*(const Tensor<T> &t){ return tensor_calculator(*this, t, MULTIPLY);}
  Tensor<T> operator/(const Tensor<T> &t){ return tensor_calculator(*this, t, DIVIDE);}
  Tensor<T> operator%(const Tensor<T> &t){ return tensor_calculator(*this, t, MOD);}

  void operator+=(const Tensor<T> &t){ *this = tensor_calculator(*this, t, PLUS);}
  void operator-=(const Tensor<T> &t){ *this = tensor_calculator(*this, t, MINUS);}
  void operator*=(const Tensor<T> &t){ *this = tensor_calculator(*this, t, MULTIPLY);}
  void operator/=(const Tensor<T> &t){ *this = tensor_calculator(*this, t, DIVIDE);}
  void operator%=(const Tensor<T> &t){ *this = tensor_calculator(*this, t, MOD);}
  void operator+=(T x){ Tensor<T> t(m_shape, x); this->operator+=(t);}
  void operator-=(T x){ Tensor<T> t(m_shape, x); this->operator-=(t);}
  void operator*=(T x){ Tensor<T> t(m_shape, x); this->operator*=(t);}
  void operator/=(T x){ Tensor<T> t(m_shape, x); this->operator/=(t);}
  void operator%=(T x){ Tensor<T> t(m_shape, x); this->operator%=(t);}

  T&      operator[](size_t idx)       { return m_data[idx];}
  const T& operator[](size_t idx) const { return m_data[idx];}

  template<typename U>
  friend std::ostream & operator<<(std::ostream &os, const Tensor<U> &t);

  Tensor<T> sum(int axis=0, bool keepdim=false){ 
    return tensor_operator(*this, axis, SUM, keepdim); }
  Tensor<T> mean(int axis=0, bool keepdim=false){
    return tensor_operator(*this, axis, MEAN, keepdim);}
  Tensor<T> max(int axis=0, bool keepdim=false){
    return tensor_operator(*this, axis, MAX, keepdim); }
  Tensor<T> min(int axis=0, bool keepdim=false){ 
    return tensor_operator(*this, axis, MIN, keepdim); }

  std::vector<T> &
  get_data(){ return m_data; }
  std::vector<T> const& 
  get_cdata() const  { return m_data; }
  std::vector<int> &
  get_shape(){ return m_shape; }
  std::vector<int> const&
  get_cshape() const { return m_shape; }

  size_t size(){ return m_data.size(); }
  size_t size() const { return m_data.size(); }

  int row(){     return m_shape[0]; }
  int col(){     return m_shape[1]; }
  int channel(){ return m_shape[2]; }  
  int number(){  return m_shape[3]; }
  int row()     const { return m_shape[0]; }
  int col()     const { return m_shape[1]; }
  int channel() const { return m_shape[2]; }  
  int number()  const { return m_shape[3]; }

  void shape(){
    printf("shape:[");
    for(int i = 0; i < m_shape.size(); i++) {
      std::cout << m_shape[i];
      if(i != m_shape.size() - 1) printf(", ");
      else printf("]\n");
    }
  }

protected:
  Tensor<T> tensor_calculator(const Tensor<T> &a, const Tensor<T> &b, int mode);
  Tensor<T> tensor_operator(Tensor<T> &t, int axis, int mode, bool keepdim);

private:
  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel [3]:number
  std::vector<T> m_data;
};


//################### Tensor::member functions' implementation ###################

  template<typename T>
  Tensor<T> 
  Tensor<T>::tensor_calculator(const Tensor<T> &a, const Tensor<T> &b, int mode){
    // col and channel must be the same shape
    assert(a.row()==b.row() && a.channel()==b.channel() && a.number()==b.number());

    Tensor<T> res(a.get_cshape());
    int ncpu = std::thread::hardware_concurrency();
    int row = a.row(), col = a.col(), channel = a.channel(), number = a.number();
#ifdef BENCH
    Timer t;
#endif
    for(int n = 0; n < number; n++){
      std::vector<std::thread> pool;
      int noffset = row * col * channel * n;
      // When a and b are totally same shape.
      if(a.row() == b.row()){
        // The channel num is way much than row, so boost for channel calculation
        if(channel >= ncpu * BOOST_CHANNEL){
          int ch_num = channel / NTHREAD_C(ncpu), ch_mod = channel % NTHREAD_C(ncpu);
          for(int i = 0; i < NTHREAD_C(ncpu); i++){
            std::thread task(vec_channel_f<T>, std::cref(a), std::cref(b), std::ref(res),
                             noffset, ch_num * i, ch_num, mode);
            pool.push_back(std::move(task));
          }
          if(ch_mod) vec_channel_f(a, b, res, noffset, channel - ch_mod, ch_mod, mode);
        }
        // The row num is way much than channel, so boost for row calculation
        else if(row >= ncpu * BOOST_ROW){
          int row_num = row / NTHREAD_R(ncpu), row_mod = row % NTHREAD_R(ncpu);
          for(int ch = 0; ch < channel; ch++){
            for(int i = 0; i < NTHREAD_C(ncpu); i++){
              std::thread task(vec_row_f<T>, std::cref(a), std::cref(b), std::ref(res),
                               noffset, ch, row_num * i, row_num, mode);
              pool.push_back(std::move(task));
            }
            if(row_mod) vec_row_f(a, b, res, noffset, ch, row - row_mod, row_mod, mode);
          }
        }
        // No need to boost
        else{
          for(int ch = 0; ch < channel; ch++) 
            vec_row_s(a, b, res, noffset, ch, 0, row, mode); 
        }
      } 
      // When a is not same shape with b.
      else{
        if(b.row() != 1) goto erro;
        
        // The channel num is way much than row, so boost for channel calculation
        if(channel >= ncpu * BOOST_CHANNEL){
          int ch_num = channel / NTHREAD_C(ncpu), ch_mod = channel % NTHREAD_C(ncpu);
          for(int i = 0; i < NTHREAD_C(ncpu); i++){
            std::thread task(vec_channel_s<T>, std::cref(a), std::cref(b), std::ref(res),
                             noffset, ch_num * i, ch_num, mode);
            pool.push_back(std::move(task));
          }
          if(ch_mod) vec_channel_s(a, b, res, noffset, channel - ch_mod, ch_mod, mode);
        } 
        // The row num is way much than channel, so boost for row calculation
        else if(row > ncpu * BOOST_ROW){
          int row_num = row / NTHREAD_R(ncpu), row_mod = row % NTHREAD_R(ncpu);
          for(int ch = 0; ch < channel; ch++){
            for(int i = 0; i < NTHREAD_C(ncpu); i++){
              std::thread task(vec_row_s<T>, std::cref(a), std::cref(b), std::ref(res),
                               noffset, ch, row_num * i, row_num, mode);
              pool.push_back(std::move(task));
            }
            if(row_mod) vec_row_s(a, b, res, noffset, ch, row - row_mod, row_mod, mode);
          }
        }
        // No need to boost
        else{
          for(int ch = 0; ch < channel; ch++) 
            vec_row_s(a, b, res, noffset, ch, 0, row, mode); 
        }
      }
      for(auto &task : pool) task.join();
    }

    return res;

  erro:
    fprintf(stderr,
    "The size of tensor a:(%d) must match the size of tensor b:(%d) \
    at non-singleton dimension 0\n", a.row(), b.row());
    exit(-1);
  }


  /* usage: operate this tensor and create a new Tensor that contain the result.
    The result's shape depend on the parameter:'mode'.
                      axis = 0:    keepdim=true      =false
    exampl:                            [[6],        (result)
      [[1, 2, 3]  -----> sum()  ----->  [15]] -----> [6, 15]
       [4, 5, 6]] -----> mean() -----> [[2],  -----> [2, 5]                     
                                        [5]] 
                  -----> max()  -----> [[3],  -----> [3, 6]
                                        [6]] 
                  -----> min()  -----> [[1],  -----> [1, 4]
                                        [4]] 
  */
  template<typename T>
  Tensor<T>
  Tensor<T>::tensor_operator(Tensor<T> &t, int axis, int mode, bool keepdim){
    int ncpu = std::thread::hardware_concurrency();
    int row = t.row(), col = t.col(), channel = t.channel(), number = t.number();
    int square = row * col, volume = square * channel;
    std::vector<std::thread> pool;
    Tensor<T> res;
#ifdef BENCH
    Timer time;
#endif
    if(axis == COL){
      if(keepdim) res = Tensor<T>(row, 1, channel, number, 0);
      else        res = Tensor<T>(1, row, channel, number, 0);
      for(int n = 0; n < number; n++){
        int noffset = volume * n, roffset = row * channel * n;
        // The channel num is way much than row, so boost for channel calculation
        if(channel >= ncpu * BOOST_CHANNEL){
          int ch_num = channel / NTHREAD_C(ncpu), ch_mod = channel % NTHREAD_C(ncpu);
          for(int i = 0; i < NTHREAD_C(ncpu); i++){
            int start = noffset + square * ch_num * i, end = start + square * ch_num;
            int res_i = roffset + ch_num * i * row;
            std::thread task(operator_axis0<T>, std::cref(t), std::ref(res), 
                            start, end, res_i, mode);
            pool.push_back(std::move(task));
          }
          if(ch_mod){
            int end = noffset + square * channel, start = end - ch_mod * square;
            int res_i = roffset + row * (channel - ch_mod);
            operator_axis0(t, res, start, end, res_i, mode);
          }
        }
        // The row num is way much than channel, so boost for row calculation.
        else if(row >= ncpu * BOOST_ROW){
          int row_num = row / NTHREAD_R(ncpu), row_mod = row % NTHREAD_R(ncpu);
          for(int ch = 0; ch < channel; ch++){
            for(int i = 0; i < NTHREAD_R(ncpu); i++){
              int start = noffset + square * ch + row_num * i * col;
              int end = start + col * row_num;
              int res_i = roffset + ch * row + row_num * i;
              std::thread task(operator_axis0<T>, std::cref(t), std::ref(res), 
                              start, end, res_i, mode);
              pool.push_back(std::move(task));
            }
            if(row_mod){
              int end = noffset + square * (ch + 1), start = end - row_mod * col;
              int res_i = roffset + (ch + 1) * row - row_mod; 
              operator_axis0(t, res, start, end, res_i, mode);
            }
          }
        }
        // Not need to boost.
        else{
          int start = noffset, end = start + volume;
          operator_axis0(t, res, start, end, roffset, mode); 
        }
      }
      for(auto &task : pool) task.join();
    }
    else if(axis == ROW){
      res = Tensor<T>(1, col, channel, number, 0);
      for(int n = 0; n < number; n++){
        int noffset = volume * n, roffset = col * channel * n;
        // The channel num is way much than row, so boost for channel calculation
        if(channel >= ncpu * BOOST_CHANNEL){
          int ch_num = channel / NTHREAD_C(ncpu), ch_mod = channel % NTHREAD_C(ncpu);
          for(int i = 0; i < NTHREAD_C(ncpu); i++){
            int start = noffset + square * ch_num * i, end = start + square * ch_num;
            int res_i = roffset + ch_num * i * col;
            std::thread task(operator_axis1_channel<T>, std::cref(t), std::ref(res), 
                            start, end, res_i, mode);
            pool.push_back(std::move(task));
          }
          if(ch_mod){
            int end = noffset + square * channel, start = end - ch_mod * square;
            int res_i = roffset + col * (channel - ch_mod);
            operator_axis1_channel(t, res, start, end, res_i, mode);
          }
        }
        // The col num is way much than row, so boost for col calculation.
        else if(col >= ncpu * BOOST_ROW){
          int col_num = col / NTHREAD_R(ncpu), col_mod = col % NTHREAD_R(ncpu);
          for(int ch = 0; ch < channel; ch++){
            for(int i = 0; i < NTHREAD_R(ncpu); i++){
              int start = noffset + square * ch + col_num * i;
              int res_i = roffset + ch * col + col_num * i;
              std::thread task(operator_axis1_col<T>, std::cref(t), std::ref(res), 
                              start, col_num, res_i, mode);
              pool.push_back(std::move(task));
            }
            if(col_mod){
              int start = noffset + square * ch + col - col_mod;
              int res_i = roffset + ch * col + col - col_mod;
              operator_axis1_col(t, res, start, col_mod, res_i, mode);
            }
          }
        }
        // Not need to boost.
        else{
          int start = noffset, end = start + volume;
          operator_axis1_channel(t, res, start, end, roffset, mode); 
        }
      }
      for(auto &task : pool) task.join();
    }
    else if(axis == CHANNEL){
      res = Tensor<T>(row, col, 1, number, 0);
      for(int n = 0; n < number; n++){
        int noffset = volume * n, roffset = row * col * n;
        // The row num is way much than col, so boost for row calculation.
        if(row >= ncpu * BOOST_ROW){
          int row_num = row / NTHREAD_R(ncpu), row_mod = row % NTHREAD_R(ncpu);
          for(int i = 0; i < NTHREAD_R(ncpu); i++){
            int start = noffset + row_num * i * col;
            int end = start + (channel - 1) * square + row_num * col;
            int res_i = roffset + start;
            std::thread task(operator_axis2_row<T>, std::cref(t), std::ref(res), 
                            start, end, row_num, res_i, mode);
            pool.push_back(std::move(task));
          }
          if(row_mod){
            int start = (row - row_mod) * col, end = t.size();
            int res_i = roffset + start;
            operator_axis2_row(t, res, start, end, row_mod, res_i, mode); 
          }
        }
        // The col num is way much than row, so boost for col calculation.
        else if(col >= ncpu * BOOST_ROW){
          int col_num = col / NTHREAD_R(ncpu), col_mod = col % NTHREAD_R(ncpu);
          for(int i = 0; i < NTHREAD_R(ncpu); i++){
            int start = noffset + col_num * i;
            int end = start + (channel - 1) * square + (row - 1) * col + col_num;
            int res_i = roffset + start;
            std::thread task(operator_axis2_col<T>, std::cref(t), std::ref(res), 
                            start, end, col_num, res_i, mode);
            pool.push_back(std::move(task));
          }
          if(col_mod){
            int start = col - col_mod, end = t.size();
            int res_i = roffset + start;
            operator_axis2_col(t, res, start, end, col_mod, res_i, mode); 
          }
        }
        // Not need to boost.
        else{
          int start = noffset, end = start + volume;
          operator_axis2_row(t, res, start, end, row, roffset, mode); 
        }
      }
      for(auto &task : pool) task.join();
    }
    return res;
  }


  template<typename U>
  std::ostream &
  operator<<(std::ostream &os, const Tensor<U> &t){
    int row = t.row(), col = t.col(), channel = t.channel(), number = t.number();
    int square = row * col, volume = square * channel;

    for(int n = 0; n < number; n++){
      int noffset = volume * n;
      if(row == 1 && channel == 1){
        printf("[");
        for(int i = 0; i < col; i++){
          os << t[noffset + i]; if(i != col - 1) os << ", ";
        }
        printf("]\n");
      }
      if(row > 1 && channel == 1){
        printf("[");
        for(int r = 0; r < row; r++){
          int row_idx = noffset + col * r;
          // printf("row_idx:%d\n", row_idx);
          if(r != 0)          putchar(' ');
          printf("[");
          for(int c = 0; c < col; c++){
            // printf("idx:%d-", row_idx + c);
            os << t[row_idx + c];
            // printf("  addr:%p", &t[row_idx + c]);
            if(c != col - 1) os << ", ";
          }
          printf("]");
          if(r != row - 1)    putchar('\n');
        }
        printf("]\n");
      }
      if(channel > 1){
        printf("[");
        for(int ch = 0; ch < channel; ch++){
          int ch_offset = noffset + ch * square;
          if(ch != 0)            printf(" ");
          printf("[");
          for(int r = 0; r < row; r++){
            int row_idx = ch_offset + col * r;
            if(r != 0)           printf("  ");
            printf("[");
            for(int c = 0; c < col; c++){
              os << t[row_idx + c];
              if(c != col - 1) os << ", ";
            }
            printf("]");
            if(r != row - 1)     printf("\n");
          }
          printf("]");
          if(ch != channel - 1)  printf(",\n");
        }
        printf("]\n");
      }
    }
    puts("");
    return os;
  }
}

