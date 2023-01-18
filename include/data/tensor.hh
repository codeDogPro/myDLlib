#pragma once

#include <data/rand_init.hh>
#include <parallel/parallel.hh>

#include <vector>
#include <numeric>
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
  Tensor(int row, int col, int channel=1, T val=-1, int number=1){
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

  Tensor<T> operator+(T x){ Tensor<T> t(m_shape, x); return calculator_invoker(t, Calculator::PLUS);}
  Tensor<T> operator-(T x){ Tensor<T> t(m_shape, x); return calculator_invoker(t, Calculator::MINUS);}
  Tensor<T> operator*(T x){ Tensor<T> t(m_shape, x); return calculator_invoker(t, Calculator::MULTIPLY);}
  Tensor<T> operator/(T x){ Tensor<T> t(m_shape, x); return calculator_invoker(t, Calculator::DIVIDE);}
  Tensor<T> operator%(T x){ Tensor<T> t(m_shape, x); return calculator_invoker(t, Calculator::MOD);}
  Tensor<T> operator+(const Tensor<T> &t){ return calculator_invoker(t, Calculator::PLUS);}
  Tensor<T> operator-(const Tensor<T> &t){ return calculator_invoker(t, Calculator::MINUS);}
  Tensor<T> operator*(const Tensor<T> &t){ return calculator_invoker(t, Calculator::MULTIPLY);}
  Tensor<T> operator/(const Tensor<T> &t){ return calculator_invoker(t, Calculator::DIVIDE);}
  Tensor<T> operator%(const Tensor<T> &t){ return calculator_invoker(t, Calculator::MOD);}

  void operator+=(const Tensor<T> &t){ *this = calculator_invoker(t, Calculator::PLUS);}
  void operator-=(const Tensor<T> &t){ *this = calculator_invoker(t, Calculator::MINUS);}
  void operator*=(const Tensor<T> &t){ *this = calculator_invoker(t, Calculator::MULTIPLY);}
  void operator/=(const Tensor<T> &t){ *this = calculator_invoker(t, Calculator::DIVIDE);}
  void operator%=(const Tensor<T> &t){ *this = calculator_invoker(t, Calculator::MOD);}
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
    return tensor_operator(*this, Axis(axis), Operator::SUM, keepdim); }
  Tensor<T> mean(int axis=0, bool keepdim=false){
    return tensor_operator(*this, Axis(axis), Operator::MEAN, keepdim);}
  Tensor<T> max(int axis=0, bool keepdim=false){
    return tensor_operator(*this, Axis(axis), Operator::MAX, keepdim); }
  Tensor<T> min(int axis=0, bool keepdim=false){ 
    return tensor_operator(*this, Axis(axis), Operator::MIN, keepdim); }

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

  int row()    { return m_shape[0]; }
  int col()    { return m_shape[1]; }
  int channel(){ return m_shape[2]; }  
  int number() { return m_shape[3]; }
  int row()     const { return m_shape[0]; }
  int col()     const { return m_shape[1]; }
  int channel() const { return m_shape[2]; }  
  int number()  const { return m_shape[3]; }

  void shape(){
    printf("shape:[");
    for(int i = 0; i < m_shape.size(); i++) {
      std::cout << m_shape[i];
      if(i != m_shape.size() - 1) printf(", ");
    }
    printf("]\n");
  }

protected:
  void tensor_calculator
  (const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &res, int num_idx, Calculator mode);

  Tensor<T> 
  calculator_invoker(const Tensor<T> &rhs, Calculator mode){
    assert(number() == rhs.number());
    Tensor<T> res(this->get_cshape(), 0);
    std::cout << res;
    for(int i = 0; i < number(); i++){
      printf("invoke time:%d\n", i + 1);
      tensor_calculator(*this, rhs, res, i, mode); 
      
      std::cout << "res:\n" << res;
    }
    return res;
  }

  Tensor<T> tensor_operator(Tensor<T> &t, Axis axis, Operator mode, bool keepdim);

private:
  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel [3]:number
  std::vector<T> m_data;
};


//################### Tensor::member functions' implementation ###################

  template<typename T>
  void Tensor<T>::tensor_calculator
  (const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &res, int num_idx, Calculator mode){
    // col and channel must be the same.
    assert(a.row() == b.row() && a.channel() == b.channel());

    int ncpu = std::thread::hardware_concurrency();
    int row = a.row(), col = a.col(), channel = a.channel();
    int noffset = row * col * channel * num_idx;
    int number = 1;

    // When a and b are totally same shape.
    if(a.row() == b.row()){
      if(channel >= ncpu * BOOST_CHANNEL){
      // boost for channel calculation
        parallel_channel(vec_channel_f<T>, 
        /*nthread, res */NTHREAD_C(ncpu, number), res,
        /*const args...*/a, b, noffset, mode);
      }
      else if(row >= ncpu * BOOST_ROW){
      // boost for row calculation
        parallel_row    (vec_row_f<T>, 
        /*nthread, res */NTHREAD_R(ncpu, number), res,
        /*const args...*/a, b, noffset, mode);
      }
      else{
      // No need to boost
        for(int ch = 0; ch < channel; ch++) 
          vec_row_f(0, row, ch, res, 
                    a, b, noffset, mode); 
      }
    } 
    // When a is not same shape with b.
    else{
      if(b.row() != 1) goto erro;
      
      if(channel >= ncpu * BOOST_CHANNEL){
      // boost for channel calculation
        parallel_channel(vec_channel_s<T>, 
        /*nthread, res */NTHREAD_C(ncpu, number), res,
        /*const args...*/a, b, noffset, mode);
      } 
      else if(row > ncpu * BOOST_ROW){
      // boost for row calculation
        parallel_row    (vec_row_s<T>, 
        /*nthread, res */NTHREAD_R(ncpu, number), res,
        /*const args...*/a, b, noffset, mode);
      }
      else{
      // No need to boost
        for(int ch = 0; ch < channel; ch++) 
          vec_row_s(0, row, ch, res, 
                    a, b, noffset, mode); 
      }
    }
    return;

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
  Tensor<T>::tensor_operator(Tensor<T> &t, Axis axis, Operator mode, bool keepdim){
    int ncpu = std::thread::hardware_concurrency();
    int row = t.row(), col = t.col(), channel = t.channel(), number = t.number();
    int square = row * col, volume = square * channel;
    Tensor<T> res;

    if(axis == Axis::COL){
      if(keepdim) res = Tensor<T>(row, 1, channel, 0, number);
      else        res = Tensor<T>(1, row, channel, 0, number);
      for(int n = 0; n < number; n++){
        int noffset = volume * n, roffset = row * channel * n;
        if(channel >= ncpu * BOOST_CHANNEL / number){
        // boost for channel calculation
          parallel_channel(operator_axis0_channel<T>,
          /*nthread, res */NTHREAD_C(ncpu, number), res,
          /*const args...*/t, noffset, roffset, mode);
        }
        else if(row >= ncpu * BOOST_ROW){
        // boost for row calculation.
          parallel_row    (operator_axis0_row<T>, 
          /*nthread, res */NTHREAD_R(ncpu, number), res,
          /*const args...*/t, noffset, roffset, mode);
        }
        else{
        // Not need to boost.
          operator_axis0_channel(0, channel, res,
                                 t, noffset, roffset, mode); 
        }
      }
    } // axix == col
    else if(axis == Axis::ROW){
      res = Tensor<T>(1, col, channel, 0, number);
      for(int n = 0; n < number; n++){
        int noffset = volume * n, roffset = col * channel * n;
        if(channel >= ncpu * BOOST_CHANNEL){
        // boost for channel calculation
          parallel_channel(operator_axis1_channel<T>, 
          /*nthread, res */NTHREAD_C(ncpu, number), res,
          /*const args...*/t, noffset, roffset, mode);
        }
        else if(row >= ncpu * BOOST_ROW){
        // boost for col calculation.
          parallel_col    (operator_axis1_col<T>, 
          /*nthread, res */NTHREAD_R(ncpu, number), res,
          /*const args...*/t, noffset, roffset, mode);
        }
        else{
        // Not need to boost.
          int start = noffset, end = start + volume;
          operator_axis1_channel(0, channel, res,
                                 t, noffset, roffset, mode); 
        }
      }
    } // axix == row
    else if(axis == Axis::CHANNEL){
      res = Tensor<T>(row, col, 1, 0, number);
      for(int n = 0; n < number; n++){
        int noffset = volume * n, roffset = row * col * n;
        if(row >= ncpu * BOOST_ROW){
        // boost for row calculation.
          parallel_row    (operator_axis2_row<T>, 
          /*nthread, res */NTHREAD_C(ncpu, number), res,
          /*const args...*/t, noffset, roffset, mode);
        }
        else if(col >= ncpu * BOOST_ROW){
        // boost for col calculation.
          parallel_col    (operator_axis2_col<T>, 
          /*nthread, res */NTHREAD_R(ncpu, number), res,
          /*const args...*/t, noffset, roffset, mode);
        }
        else{
        // Not need to boost.
          operator_axis2_row(0, row, channel, res,
                             t, noffset, roffset, mode); 
        }
      }
    } // axis == channel
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

} // namespace dl

