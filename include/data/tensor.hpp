#pragma once

#define BENCH

#include <data/rand_init.hpp>
#include <parallel/parallel.hpp>
#ifdef BENCH
#include <basic/timer.hpp>
#endif

#include <vector>
#include <numeric>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <assert.h>

namespace dl{

template<typename T>
class Tensor{

public:
  Tensor() = delete;

  Tensor(int row, int col, int channel=1, T val=0){
    assert(row != 0 && col != 0 && channel != 0);

    m_data.assign(row * col * channel, val);
    m_shape.assign({row, col, channel});
    if(val == 0){ rand_init(*this);}
  }

  Tensor(const std::vector<int> &shape, T val=0){
    assert(shape.size() != 0);

    int product = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    m_data.assign(product, val);
    m_shape = shape;
    if(val == 0){ rand_init(*this);}
  }

  Tensor(std::vector<int> &data, std::vector<int> &shape)
  : m_data(data), m_shape(shape){}

  // deep copy
  Tensor(const Tensor<T> &t){ 
    m_data.assign(t.m_data.size(), 0);
    m_shape = t.m_shape;
    /*TODO: deep copy*/
  }

  // move copy
  Tensor(Tensor<T> &&t){ 
    m_data  = std::move(t.m_data);
    m_shape = std::move(t.m_shape);
  }

  Tensor<T> &
  operator=(const Tensor<T> &t){
    m_data.assign(t.m_data.size(), 0);
    m_shape = t.m_shape;
    return *this;
  }

  Tensor<T> &
  operator=(Tensor<T> &&t){
    m_data  = std::move(t.m_data);
    m_shape = std::move(t.m_shape);
    return *this;
  }

  Tensor<T> calculator(Tensor<T> &a, Tensor<T> &b, int mode);

  Tensor<T> 
  operator+(Tensor<T> &t){ return calculator(*this, t, PLUS);}
  Tensor<T> 
  operator+(T x){ Tensor<T> t(m_shape, x); return calculator(*this, t, PLUS);}
  Tensor<T> 
  operator-(Tensor<T> &t){ return calculator(*this, t, MINUS);}
  Tensor<T> 
  operator-(T x){ Tensor<T> t(m_shape, x); return calculator(*this, t, MINUS);}
  Tensor<T> 
  operator*(Tensor<T> &t){ return calculator(*this, t, MULTIPLY);}
  Tensor<T> 
  operator*(T x){ Tensor<T> t(m_shape, x); return calculator(*this, t, MULTIPLY);}
  Tensor<T> 
  operator/(Tensor<T> &t){ return calculator(*this, t, DIVIDE);}
  Tensor<T> 
  operator/(T x){ Tensor<T> t(m_shape, x); return calculator(*this, t, DIVIDE);}
  Tensor<T> 
  operator%(Tensor<T> &t){ return calculator(*this, t, MOD);}
  Tensor<T> 
  operator%(T x){ Tensor<T> t(m_shape, x); return calculator(*this, t, MOD);}

  void operator+=(Tensor<T> &t){ *this = calculator(*this, t, PLUS);}
  void operator-=(Tensor<T> &t){ *this = calculator(*this, t, MINUS);}
  void operator*=(Tensor<T> &t){ *this = calculator(*this, t, MULTIPLY);}
  void operator/=(Tensor<T> &t){ *this = calculator(*this, t, DIVIDE);}
  void operator%=(Tensor<T> &t){ *this = calculator(*this, t, MOD);}
  void operator+=(T x){ Tensor<T> t(m_shape, x); this->operator+=(t);}
  void operator-=(T x){ Tensor<T> t(m_shape, x); this->operator-=(t);}
  void operator*=(T x){ Tensor<T> t(m_shape, x); this->operator*=(t);}
  void operator/=(T x){ Tensor<T> t(m_shape, x); this->operator/=(t);}
  void operator%=(T x){ Tensor<T> t(m_shape, x); this->operator%=(t);}

  T&   operator[](int idx){ return m_data[idx];}

  template<typename U>
  friend std::ostream & operator<<(std::ostream &os, const Tensor<U> &t);

  Tensor<T> sum(int axis=0, bool keepdim=false);
  Tensor<T> mean(int axis=0, bool keepdim=false);
  Tensor<T> max(int axis=0, bool keepdim=false);
  Tensor<T> min(int axis=0, bool keepdim=false);

  void shape(){
    printf("shape:[");
    for(int i = 0; i < m_shape.size(); i++) {
      std::cout << m_shape[i];
      if(i != m_shape.size() - 1) printf(", ");
      else printf("]\n");
    }
  }

  std::vector<T> &
  get_data(){ return m_data; }

  const std::vector<T> & 
  get_cdata(){ return m_data; }

  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel

private:
  std::vector<T> m_data;
};

//################### Tensor::member functions' implementation ###################


  template<typename T>
  Tensor<T> 
  Tensor<T>::calculator(Tensor<T> &a, Tensor<T> &b, int mode){
    // col and channel must be the same shape
    assert(a.m_shape[1] == b.m_shape[1] && a.m_shape[2] == b.m_shape[2]);

    Tensor<T> result(a.m_shape);
    int ncpu = std::thread::hardware_concurrency();
    int row = a.m_shape[0], col = a.m_shape[1], channel = a.m_shape[2];
#ifdef BENCH
    Timer t;
#endif
    std::vector<std::thread> pool;
    // When a and b are totally same shape.
    if(a.m_shape[0] == b.m_shape[0]){
      // The channel num is way much than row, so boost for channel calculation
      if(channel >= ncpu * BOOST_CHANNEL){
        int ch_num = NTHREAD_C(ncpu) / channel;
        for(int i = 0; i < NTHREAD_C(ncpu); i++){
          std::thread task(vec_channel_f<T>, std::ref(a), std::ref(b), std::ref(result),
                           ch_num * i, ch_num, mode);
          pool.push_back(std::move(task));
        }
      } else{
      // The row num is way much than channel, so boost for row calculation
        if(row >= ncpu * BOOST_ROW){
          int row_num = NTHREAD_R(ncpu) / row;
          for(int ch = 0; ch < channel; ch++)
            for(int i = 0; i < NTHREAD_C(ncpu); i++){
              std::thread task(vec_row_f<T>, std::ref(a), std::ref(b), std::ref(result),
                              ch, row_num * i, row_num, mode);
              pool.push_back(std::move(task));
            }
        } else{
          for(int ch = 0; ch < channel; ch++)
            vec_row_s(a, b, result, ch, 0, row, mode); 
        }
      }
    } 
    // When a is not same shape with b.
    else{
      if(b.m_shape[0] != 1) goto erro;
      
      if(channel >= ncpu * BOOST_CHANNEL){
        int ch_num = NTHREAD_C(ncpu) / channel;
        for(int i = 0; i < NTHREAD_C(ncpu); i++){
          std::thread task(vec_channel_s<T>, std::ref(a), std::ref(b), std::ref(result),
                           ch_num * i, ch_num, mode);
          pool.push_back(std::move(task));
        }
      } else{
        if(row > ncpu * BOOST_ROW){
          int row_num = NTHREAD_R(ncpu) / row;
          for(int ch = 0; ch < channel; ch++)
            for(int i = 0; i < NTHREAD_C(ncpu); i++){
              std::thread task(vec_row_s<T>, std::ref(a), std::ref(b), std::ref(result),
                              ch, row_num * i, row_num, mode);
              pool.push_back(std::move(task));
            }
        } else{
          for(int ch = 0; ch < channel; ch++)
            vec_row_s(a, b, result, ch, 0, row, mode); 
        }
      }
    }
    for(auto &task : pool) task.join();

    return result;

  erro:
    fprintf(stderr,
    "The size of tensor a:(%d) must match the size of tensor b:(%d) \
    at non-singleton dimension 0\n", a.m_shape[0], b.m_shape[0]);
    exit(-1);
  }

  /*
    usage: Sum up each row, and create a new Tensor that contain the result.
    The result's shape must be [1 x out_dim x channel]

    exampl:                            [[6],        (result)
      [[1, 2, 3]  -----> sum()  ----->  [15]] -----> [6, 15]
       [4, 5, 6]] -----> mean() -----> [[2],  -----> [2, 5]                     
                                        [5]]
  */
  template<typename T>
  static Tensor<T>
  sum_or_mean(Tensor<T> &t, int mode){
    int row = t.m_shape[0], col = t.m_shape[1], channel = t.m_shape[2];
    std::vector<T> data, shape{1, row, channel};
    for(int cnt = 0, sum = 0; T x : t.get_cdata()){
      sum += x;
      if(++cnt == col){
        if(mode == SUM) 
          data.push_back(sum);
        if(mode == MEAN)
          data.push_back(sum / col);
        sum = cnt = 0;
      }
    }
    assert(data.size() == row * channel);
    return Tensor<T>(data, shape);
  }

  template<typename T>
  Tensor<T> 
  Tensor<T>::sum(int axis=0, bool keepdim=false){
    return sum_or_means<T>(*this, SUM);
  }
  template<typename T>
  Tensor<T> 
  Tensor<T>::mean(int axis=0, bool keepdim=false){
    return sum_or_means<T>(*this, MEAN);
  }


  template<typename U>
  std::ostream &
  operator<<(std::ostream &os, const Tensor<U> &t){
    int height = t.m_shape[0], width = t.m_shape[1];
    if(t.m_shape[0] == 1){
      printf("[");
      for(int i = 0; i < t.m_data.size(); i++){
        os << t.m_data[i];
        if(i != t.m_data.size() - 1) os << ", ";
      }
      printf("]\n");
    }
    if(t.m_shape[0] > 1 && t.m_shape[2] == 1){
      printf("[");
      for(int h = 0; h < height; h++){
        int row_idx = h * width;
        if(h != 0)          putchar(' ');
        printf("[");
        for(int w = 0; w < width; w++){
          os << t.m_data[row_idx + w];
          if(w != width - 1) os << ", ";
        }
        printf("]");
        if(h != height - 1) putchar('\n');
      }
      printf("]\n");
    }
    if(t.m_shape[2] > 1){
      int channel = t.m_shape[2];
      printf("[");
      for(int c = 0; c < channel; c++){
        printf("[");
        for(int h = 0; h < height; h++){
          int row_idx = h * width;
          printf("[");
          for(int w = 0; w < width; w++){
            os << t.m_data[row_idx + w];
            if(w != width - 1) os << ", ";
          }
          printf("]");
          if(h != height - 1) printf("\n");
        }
        printf("]");
        if(c != channel - 1) printf(",\n");
      }
      printf("]\n");
    }
    puts("");
    return os;
  }
}

