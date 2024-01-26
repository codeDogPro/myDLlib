#pragma once

#include <data/rand_init.hh>
#include <data/align_alloc.hh>
#include <parallel/parallel.hh>
#include <parallel/tensor_parallel.hh>

#include <numeric>
#include <cstdlib>
#include <cstring>
#include <iomanip>

#include <opencv2/core.hpp>

namespace dl{

template<typename T=f32>
class Tensor{

public:
  explicit Tensor() = default;

  explicit
  Tensor(int row, int col, int channel=1, int number=1, T val=T(-1)){
    assert(row != 0 && col != 0 && channel != 0 && number != 0);

    m_data.assign(row * col * channel * number, val);
    m_shape.assign({row, col, channel, number});
    full_print = false;
    if(val == T(-1)){ rand_init(*this);}
  }

  explicit
  Tensor(const std::vector<int>& shape, T val=-1){
    assert(shape.size() != 0);

    int size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    m_data.assign(size, val);
    m_shape = shape;
    full_print = false;
    if(val == T(-1)){ rand_init(*this);}
  }

  explicit
  Tensor(std::vector<T, AlignedAllocator<T, 64>>& data, std::vector<int>& shape)
  : m_data(data), m_shape(shape), full_print(false) {}

  // deep copy
  explicit
  Tensor(const Tensor<T>& t){ 
    m_shape.assign(3, 0); m_data.assign(t.size(), 0);
    for(int i = 0; int x : t.get_cshape()) m_shape[i++] = x;
    parallel_copy(t);
    full_print = false;
  }

  // move copy ctor
  Tensor(Tensor<T>&& t){ 
    m_data  = t.get_data();
    m_shape = t.get_shape();
    full_print = false;
  }

  Tensor<T> &
  operator=(const Tensor<T>& rhs){
    if(this == &rhs) return *this;

    // puts("invoke operator= copy");
    m_shape.assign(4, 0); m_data.assign(rhs.size(), 0);
    for(int i = 0; int x : rhs.get_cshape()) m_shape[i++] = x;

    parallel_copy(rhs);
    // puts("Finish operator= copy");
    full_print = false;
    return *this;
  }

  Tensor<T> &
  operator=(Tensor<T>&& rhs){
    if(this == &rhs) return *this;

    puts("invoke operator= move");
    m_data  = std::move(rhs.get_data());
    m_shape = std::move(rhs.get_shape());
    full_print = false;
    return *this;
  }

  std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& rhs) { return tensor_calculator(*this, rhs, Calculator::ADD);}
  std::shared_ptr<Tensor<T>> operator-(const Tensor<T>& rhs) { return tensor_calculator(*this, rhs, Calculator::SUB);}
  std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& rhs) { return tensor_calculator(*this, rhs, Calculator::MUL);}
  std::shared_ptr<Tensor<T>> operator/(const Tensor<T>& rhs) { return tensor_calculator(*this, rhs, Calculator::DIV);}
  std::shared_ptr<Tensor<T>> operator+(T x) { Tensor<T> rhs(m_shape, x); return *this + rhs; }
  std::shared_ptr<Tensor<T>> operator-(T x) { Tensor<T> rhs(m_shape, x); return *this - rhs; }
  std::shared_ptr<Tensor<T>> operator*(T x) { Tensor<T> rhs(m_shape, x); return *this * rhs; }
  std::shared_ptr<Tensor<T>> operator/(T x) { Tensor<T> rhs(m_shape, x); return *this / rhs; }

  void operator+=(const Tensor<T>& rhs) { *this = *(*this + rhs); }
  void operator-=(const Tensor<T>& rhs) { *this = *(*this - rhs); }
  void operator*=(const Tensor<T>& rhs) { *this = *(*this * rhs); }
  void operator/=(const Tensor<T>& rhs) { *this = *(*this / rhs); }
  void operator+=(T x) { Tensor<T> rhs(m_shape, x); *this += rhs; }
  void operator-=(T x) { Tensor<T> rhs(m_shape, x); *this -= rhs; }
  void operator*=(T x) { Tensor<T> rhs(m_shape, x); *this *= rhs; }
  void operator/=(T x) { Tensor<T> rhs(m_shape, x); *this /= rhs; }

        T& operator[](size_t idx)       { return m_data[idx];}
  const T& operator[](size_t idx) const { return m_data[idx];}

  template<typename U>
  friend std::ostream & operator<<(std::ostream &os, const Tensor<U> &t);

  std::shared_ptr<Tensor<T>>
  sum(int axis=0, bool keepdim=false){ 
    return tensor_operator(Axis(axis), Operator::SUM, keepdim);
  }
  std::shared_ptr<Tensor<T>>
  mean(int axis=0, bool keepdim=false){
    return tensor_operator(Axis(axis), Operator::MEAN, keepdim);
  }
  std::shared_ptr<Tensor<T>>
  max(int axis=0, bool keepdim=false){
    return tensor_operator(Axis(axis), Operator::MAX, keepdim);
  }
  std::shared_ptr<Tensor<T>>
  min(int axis=0, bool keepdim=false){ 
    return tensor_operator(Axis(axis), Operator::MIN, keepdim);
  }

  T * const               data()       const {return m_data.data();} 
  T * const               data()             {return m_data.data();} 
  std::vector<int> &      get_shape ()       { return m_shape;}
  std::vector<int> const& get_cshape() const { return m_shape;}
  std::vector<T, AlignedAllocator<T, 64>> &      
  get_data ()       { return m_data; }
  std::vector<T, AlignedAllocator<T, 64>> const& 
  get_cdata() const { return m_data; }

  bool reshape(const std::vector<int>& shape){ 
    size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    if(size != m_data.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_data.size());
      exit(-1);
    }
    m_shape = shape;
    return true;
  } 
  bool reshape(int row, int col, int channel, int number=1){
    size_t size = row * col * channel * number;
    if(size != m_data.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_data.size());
      exit(-1);
    }
    m_shape[0] = row, m_shape[1] = col, m_shape[2] = channel, m_shape[3] = number;
    return true;
  } 


  size_t size()       { return m_data.size(); }
  size_t size() const { return m_data.size(); }

  int row()           { return m_shape[0]; }
  int col()           { return m_shape[1]; }
  int channel()       { return m_shape[2]; }  
  int number()        { return m_shape[3]; }
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

  void setFullPrintMode(bool mode)       { full_print = mode;}
  void setFullPrintMode(bool mode) const { full_print = mode;}
  bool getFullPrintMode()       { return full_print;}
  bool getFullPrintMode() const { return full_print;}

protected:
  void parallel_copy(const Tensor<T> &rhs);
  std::shared_ptr<Tensor<T>> tensor_calculator(const Tensor<T>& lhs, const Tensor<T>& rhs, Calculator mode);
  std::shared_ptr<Tensor<T>> tensor_operator(Axis axis, Operator mode, bool keepdim);

private:
  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel [3]:number
  std::vector<T, AlignedAllocator<T, 64>> m_data;
  bool full_print;
}; // class Tensor


//################### Tensor::member functions' implementation ###################

  template<typename T>
  void Tensor<T>::parallel_copy(const Tensor<T> &rhs){
    int row = this->row(), col = this->col();
    int number = this->number(), channel = this->channel();
    // insure lhs won't be deleted when finish copy
    auto deleter = [](Tensor<T> *tensor){ };
    std::shared_ptr<Tensor<T>> lhs(this, deleter);
    if(number >= BOOST_NUMBER){
      parallelizer.parallel_number(tensor_copy<T>, lhs, rhs);
    }
    else{
      for(int i = 0; i < number; i++){
        int offset = i * row * col * channel;
        parallelizer.parallel_channel(tensor_copy<T>, lhs, offset, rhs);
      }
    }
    parallelizer.sync();
  }


  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::tensor_calculator
  (const Tensor<T>& lhs, const Tensor<T>& rhs, Calculator mode){
    // col and channel and number must be the same.
    assert(lhs.col() == rhs.col() && lhs.channel() == rhs.channel() &&
           lhs.number() == rhs.number());
    int lrow = lhs.row(), rrow = rhs.row(); 
    int col = lhs.col(), channel = lhs.channel(), number = lhs.number();

    std::shared_ptr<Tensor<T>> output;
    if(lrow > rrow){
      if(rrow == 1){
        int volume = lrow * col * channel;
        output = std::make_shared<Tensor<T>>(lhs.get_cshape(), 0);
        if(channel >= number * BOOST_CHANNEL){
          for(int i = 0; i < number; i++){
            int offset = i * volume;
            switch(mode){
              case Calculator::ADD:
                parallelizer.parallel_channel(
                  vec_add_single<T>, output, offset, lhs, rhs); break;
              case Calculator::SUB:
                parallelizer.parallel_channel(
                  vec_sub_single<T>, output, offset, lhs, rhs); break;
              case Calculator::MUL:
                parallelizer.parallel_channel(
                  vec_mul_single<T>, output, offset, lhs, rhs); break;
              case Calculator::DIV:
                parallelizer.parallel_channel(
                  vec_div_single<T>, output, offset, lhs, rhs); break;
              default: assert(0); 
            }
          }
        }
        else{
          switch(mode){
            case Calculator::ADD:
              parallelizer.parallel_number(
                vec_add_single<T>, output, lhs, rhs); break;
            case Calculator::SUB:
              parallelizer.parallel_number(
                vec_sub_single<T>, output, lhs, rhs); break;
            case Calculator::MUL:
              parallelizer.parallel_number(
                vec_mul_single<T>, output, lhs, rhs); break;
            case Calculator::DIV:
              parallelizer.parallel_number(
                vec_div_single<T>, output, lhs, rhs); break;
            default: assert(0); 
          }
        }
      }
      else goto erro;
    }
    else if(lrow == rrow){
      int volume = lrow * col * channel;
      output = std::make_shared<Tensor<T>>(lhs.get_cshape(), 0);
      if(channel >= number * BOOST_CHANNEL){
        for(int i = 0; i < number; i++){
          int offset = i * volume;
          switch(mode){
            case Calculator::ADD:
              parallelizer.parallel_channel(
                vec_add_full<T>, output, offset, lhs, rhs); break;
            case Calculator::SUB:
              parallelizer.parallel_channel(
                vec_sub_full<T>, output, offset, lhs, rhs); break;
            case Calculator::MUL:
              parallelizer.parallel_channel(
                vec_mul_full<T>, output, offset, lhs, rhs); break;
            case Calculator::DIV:
              parallelizer.parallel_channel(
                vec_div_full<T>, output, offset, lhs, rhs); break;
            default: assert(0); 
          }
        }
      }
      else {
        switch(mode){
          case Calculator::ADD:
            parallelizer.parallel_number(
              vec_add_full<T>, output, lhs, rhs); break;
          case Calculator::SUB:
            parallelizer.parallel_number(
              vec_sub_full<T>, output, lhs, rhs); break;
          case Calculator::MUL:
            parallelizer.parallel_number(
              vec_mul_full<T>, output, lhs, rhs); break;
          case Calculator::DIV:
            parallelizer.parallel_number(
              vec_div_full<T>, output, lhs, rhs); break;
          default: assert(0); 
        }
      }
    }
    else{ // lhs.row < rhs.row
      if(lrow == 1){
        // puts("lhs row == 1");
        int volume = rrow * col * channel;
        output = std::make_shared<Tensor<T>>(rhs.get_cshape(), 0);
        if(channel >= number * BOOST_CHANNEL){
          for(int i = 0; i < number; i++){
            int offset = i * volume;
            switch(mode){
              case Calculator::ADD:
                parallelizer.parallel_channel(
                  vec_add_single<T>, output, offset, rhs, *this); break;
              case Calculator::SUB:
                parallelizer.parallel_channel(
                  vec_sub_single<T>, output, offset, rhs, *this); break;
              case Calculator::MUL:
                parallelizer.parallel_channel(
                  vec_mul_single<T>, output, offset, rhs, *this); break;
              case Calculator::DIV:
                parallelizer.parallel_channel(
                  vec_div_single<T>, output, offset, rhs, *this); break;
              default: assert(0); 
            }
          }
        }
        else{
          switch(mode){
            case Calculator::ADD:
              parallelizer.parallel_number(
                vec_add_single<T>, output, rhs, *this); break;
            case Calculator::SUB:
              parallelizer.parallel_number(
                vec_sub_single<T>, output, rhs, *this); break;
            case Calculator::MUL:
              parallelizer.parallel_number(
                vec_mul_single<T>, output, rhs, *this); break;
            case Calculator::DIV:
              parallelizer.parallel_number(
                vec_div_single<T>, output, rhs, *this); break;
            default: assert(0); 
          }
        }
      }
      else goto erro;
    }
    parallelizer.sync();
    return output;

  erro:
    fprintf(stderr, "The size of tensor lhs:(%d) must match the size of \
    tensor rhs:(%d) at non-singleton dimension 0\n", lhs.row(), rhs.row());
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
                                        [4]] */
  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::tensor_operator(Axis axis, Operator mode, bool keepdim){
    int row = this->row(), col = this->col(), channel = this->channel();
    int square = row * col, volume = square * channel;
    int number = this->number();
    std::shared_ptr<Tensor<T>> output;
    if(axis == Axis::COL){
      if(keepdim == true){
        output = std::make_shared<Tensor<T>>(row, 1, channel, number, 0);
      }
      else{
        output = std::make_shared<Tensor<T>>(1, row, channel, number, 0);
      }
      for(int i = 0; i < number; i++){
        int offset = i * volume;
        switch(mode){
          case Operator::SUM:
            parallelizer.parallel_channel(
              operator_sum_axis0<T>, output, offset, *this); break;
          case Operator::MEAN:
            parallelizer.parallel_channel(
              operator_mean_axis0<T>, output, offset, *this); break;
          case Operator::MAX:
            parallelizer.parallel_channel(
              operator_max_axis0<T>, output, offset, *this); break;
          case Operator::MIN:
            parallelizer.parallel_channel(
              operator_min_axis0<T>, output, offset, *this); break;
        }
      }
    }
    else if(axis == Axis::ROW){
      output = std::make_shared<Tensor<T>>(1, col, channel, number, 0);
      for(int i = 0; i < number; i++){
        int offset = i * volume;
        switch(mode){
          case Operator::SUM:
            parallelizer.parallel_channel(
              operator_sum_axis1<T>, output, offset, *this); break;
          case Operator::MEAN:
            parallelizer.parallel_channel(
              operator_mean_axis1<T>, output, offset, *this); break;
          case Operator::MAX:
            parallelizer.parallel_channel(
              operator_max_axis1<T>, output, offset, *this); break;
          case Operator::MIN:
            parallelizer.parallel_channel(
              operator_min_axis1<T>, output, offset, *this); break;
        }
      }
    }
    else if(axis == Axis::CHANNEL){
      if(keepdim == true){
        output = std::make_shared<Tensor<T>>(row, col, number, 1, 0);
      }
      else{
        output = std::make_shared<Tensor<T>>(row, col, 1, number, 0);
      }
      for(int i = 0; i < number; i++){
        int offset = i * volume;
        switch(mode){
          case Operator::SUM:
            parallelizer.parallel_row(
              operator_sum_axis2<T>, output, offset, *this); break;
          case Operator::MEAN:
            parallelizer.parallel_row(
              operator_mean_axis2<T>, output, offset, *this); break;
          case Operator::MAX:
            parallelizer.parallel_row(
              operator_max_axis2<T>, output, offset, *this); break;
          case Operator::MIN:
            parallelizer.parallel_row(
              operator_min_axis2<T>, output, offset, *this); break;
        }
      }
    }
    parallelizer.sync();
    return output;
  }
  
  
  // ######################################### PRINT ##################################################
  template<typename T>
  void print_H(std::ostream &os, const Tensor<T> &t, int offset){
    using std::setw;
    // os.setf(std::ios::scientific);  // 科学计数法
    os.precision(PRINT_PRECISION);

    int col = t.col();
    if(col > MAX_PRINT_COL){
      if(t.getFullPrintMode() == true){
        for(int i = 0; i < col; i++){
          os << setw(7) << t[offset + i]; if(i != col - 1) os << ", ";
        }
      }
      else{ // 省略输出模式
        for(int i = 0; i < SHOW_NUMBER_LEN; i++){
          os << setw(7) << t[offset + i] << ", ";
        }
        printf(" ..., ");
        for(int i = col - SHOW_NUMBER_LEN; i < col; i++){
          os << setw(7) << t[offset + i]; if(i != col - 1) os << ", ";
        }
      }
    }
    else{
      for(int i = 0; i < col; i++){
        os << setw(7) << t[offset + i]; if(i != col - 1) os << ", ";
      }
    }
  }

  template<typename T>
  void print_WxH(std::ostream &os, const Tensor<T> &t, int offset){
    int row = t.row(), col = t.col(), number = t.number();
    if(row > MAX_PRINT_ROW){
      if(t.getFullPrintMode() == true){
        for(int r = 0; r < row; r++){
          int row_idx = offset + col * r;
          if(number > 1 && r != 0) printf(" ");
          if(r != 0)               printf("  ");
          printf("[");
          print_H(os, t, row_idx);
          printf("]");
          if(r != row - 1)         printf(",\n");
        }
      }
      else{ // 省略输出模式
        for(int r = 0; r < SHOW_NUMBER_LEN; r++){
          int row_idx = offset + col * r;
          if(number > 1 && r != 0) printf(" ");
          if(r != 0)               printf("  ");
          printf("[");
          print_H(os, t, row_idx);
          printf("],\n");
        }
        if(number > 1) printf("   ...,\n");
        else           printf("  ...,\n");
        for(int r = row - SHOW_NUMBER_LEN; r < row; r++){
          int row_idx = offset + col * r;
          if(number > 1 && r != 0) printf(" ");
          printf("  [");
          print_H(os, t, row_idx);
          printf("]");
          if(r != row - 1)         printf(",\n");
        }
      }
    }
    else{
      for(int r = 0; r < row; r++){
        int row_idx = offset + col * r;
        if(number > 1 && r != 0) printf(" ");
        if(r != 0)               printf("  ");
        printf("[");
        print_H(os, t, row_idx);
        printf("]");
        if(r != row - 1)         printf(",\n");
      }
    }
  }

  template<typename T>
  void print_CxWxH(std::ostream &os, const Tensor<T> &t, int offset){
    int row = t.row(), col = t.col(), channel = t.channel(), number = t.number();
    if(channel > MAX_PRINT_CHANNEL){
      if(t.getFullPrintMode() == true){
        for(int ch = 0; ch < channel; ch++){
          int ch_offset = offset + ch * row * col;
          if(number > 1 && ch != 0)  printf(" ");
          if(ch != 0)                printf(" ");
          printf("[");
          print_WxH(os, t, ch_offset);
          printf("]");
          if(ch != channel - 1)      printf(",\n");
        }
      }
      else{ // 省略输出模式
        for(int ch = 0; ch < SHOW_NUMBER_LEN; ch++){
          int ch_offset = offset + ch * row * col;
          if(number > 1 && ch != 0)  printf(" ");
          if(ch != 0)                printf(" ");
          printf("[");
          print_WxH(os, t, ch_offset);
          printf("],\n");
        }
        if(number > 1) printf("\n  ...,\n\n");
        else           printf("\n ...,\n\n");
        for(int ch = channel - SHOW_NUMBER_LEN; ch < channel; ch++){
          int ch_offset = offset + ch * row * col;
          if(number > 1 && ch != 0)  printf(" ");
          printf(" [");
          print_WxH(os, t, ch_offset);
          printf("]");
          if(ch != channel - 1)      printf(",\n");
        }
      }
    }
    else{
      for(int ch = 0; ch < channel; ch++){
        int ch_offset = offset + ch * row * col;
        if(number > 1 && ch != 0)  printf(" ");
        if(ch != 0)                printf(" ");
        printf("[");
        print_WxH(os, t, ch_offset);
        printf("]");
        if(ch != channel - 1)      printf(",\n");
      }
    }
  }

  template<typename U>
  std::ostream &
  operator<<(std::ostream& os, const Tensor<U>& t){
    int row = t.row(), col = t.col(), channel = t.channel(), number = t.number();
    int square = row * col, volume = square * channel;

    if(number > 1)   printf("[");
    for(int n = 0; n < number; n++){
      int noffset = volume * n;
      if(n != 0) printf(" ");
      if(row == 1 && channel == 1){
        printf("[");
        print_H(os, t, noffset);
        if(number == 1)            printf("]\n"); 
        else if(n != number - 1)   printf("],\n");
        else if(n == number - 1)   printf("]");
      }
      else if(row > 1 && channel == 1){
        printf("[");
        print_WxH(os, t, noffset);
        if(number == 1)            printf("]\n"); 
        else if(n != number - 1)   printf("],\n");
        else if(n == number - 1)   printf("]");
      }
      else if(channel > 1){
        printf("[");
        print_CxWxH(os, t, noffset);
        if(number == 1)              printf("]\n"); 
        else if(n != number - 1)     printf("],\n\n");
        else if(n == number - 1)     printf("]");
      }
    }
    if(number > 1) printf("]\n\n");
    return os;
  }
  // ########################################### END PRINT ###################################################



  template<typename T=f32>
  std::shared_ptr<Tensor<T>>
  to_Tensor(const cv::Mat &data){
    cv::MatSize size = data.size;
    int channel = data.channels();
    auto output = std::make_shared<Tensor<T>>(size[0], size[1], channel, 1, 0);
    /*
    because the multichannels mat's memory struct, 
    we need to boost base on row, and each thread
    handle all the channel.
    */
    parallelizer.parallel_row(cvMat2Tensor<T>, output, 0, data);
    parallelizer.sync();
    return output;
  }
} // namespace dl

