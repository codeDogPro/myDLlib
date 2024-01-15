#pragma once

#include <data/rand_init.hh>
#include <parallel/parallel.hh>
#include <parallel/tensor_parallel.hh>

#include <numeric>
#include <cstdlib>
#include <cstring>
#include <iomanip>

namespace dl{

template<typename T=f32>
class Tensor{

public:
  // ~Tensor() {
  //   puts("Tensor destroyed!");
  // }
  explicit Tensor() = default;

  explicit
  Tensor(int row, int col, int channel=1, int number=1, T val=T(-1)){
    assert(row != 0 && col != 0 && channel != 0 && number != 0);

    // std::cout << "type:" << type_name<T>() << "   val:" << val << std::endl;
    m_data.assign(row * col * channel * number, val);
    m_shape.assign({row, col, channel, number});
    if(val == T(-1)){ rand_init(*this);}
  }

  explicit
  Tensor(const std::vector<int>& shape, T val=-1){
    assert(shape.size() != 0);

    int size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    m_data.assign(size, val);
    m_shape = shape;
    if(val == -1){ rand_init(*this);}
  }

  explicit
  Tensor(std::vector<int>& data, std::vector<int>& shape)
  : m_data(data), m_shape(shape){}

  // deep copy
  explicit
  Tensor(const Tensor<T>& t){ 
    m_shape.assign(3, 0); m_data.assign(t.size(), 0);
    for(int i = 0; int x : t.get_cshape()) m_shape[i++] = x;
    parallel_copy(t);
  }

  // move copy ctor
  Tensor(Tensor<T>&& t){ 
    m_data  = t.get_data();
    m_shape = t.get_shape();
  }

  Tensor<T> &
  operator=(const Tensor<T>& rhs){
    if(this == &rhs) return *this;

    puts("invoke operator= copy");
    m_shape.assign(4, 0); m_data.assign(rhs.size(), 0);
    for(int i = 0; int x : rhs.get_cshape()) m_shape[i++] = x;

    parallel_copy(rhs);
    puts("Finish operator= copy");
    // for(int i = 0; int x : rhs.get_cdata()) m_data[i++] = x;
    return *this;
  }

  Tensor<T> &
  operator=(Tensor<T>&& rhs){
    if(this == &rhs) return *this;

    puts("invoke operator= move");
    m_data  = std::move(rhs.get_data());
    m_shape = std::move(rhs.get_shape());
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
  std::vector<T>   &      get_data  ()       { return m_data; }
  std::vector<int> &      get_shape ()       { return m_shape;}
  std::vector<T>   const& get_cdata () const { return m_data; }
  std::vector<int> const& get_cshape() const { return m_shape;}

  bool reshape(const std::vector<int>& shape){ 
    size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    if(size != m_data.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_data.size());
      exit(-1);
    }
    m_shape = shape;
    if(m_data.size() == size) return true;
    return false; 
  } 
  bool reshape(int row, int col, int channel, int number=1){
    m_shape[0] = row, m_shape[1] = col, m_shape[2] = channel, m_shape[3] = number;
    size_t size = row * col * channel * number;
    if(size != m_data.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_data.size());
      exit(-1);
    }
    m_shape = shape;
    if(m_data.size() == size) return true;
    return false; 
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

protected:
  void parallel_copy(const Tensor<T> &rhs);
  std::shared_ptr<Tensor<T>>
  tensor_calculator(const Tensor<T>& lhs, const Tensor<T>& rhs, Calculator mode);
  std::shared_ptr<Tensor<T>>
  tensor_operator(Axis axis, Operator mode, bool keepdim);

private:
  std::vector<int> m_shape; // [0]:row [1]:col [2]:channel [3]:number
  std::vector<T> m_data;
};


//################### Tensor::member functions' implementation ###################

  template<typename T>
  void Tensor<T>::parallel_copy(const Tensor<T> &rhs){
    int row = this->row(), col = this->col();
    int number = this->number(), channel = this->channel();
    auto deleter = [](Tensor<T> *tensor){ };
    std::shared_ptr<Tensor<T>> lhs(this, deleter);
    if(channel >= number * BOOST_CHANNEL){
      for(int i = 0; i < number; i++){
        int offset = i * row * col * channel;
        parallelizer.parallel_channel(tensor_copy<T>, lhs, offset, rhs);
      }
    }
    else {
      parallelizer.parallel_number(tensor_copy<T>, lhs, rhs);
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
    // std::cout << (*output) << std::endl;
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
    int ncpu = cpu_number();
    int row = this->row(), col = this->col(), channel = this->channel();
    int number = this->number();
    int square = row * col, volume = square * channel;
    return nullptr;

    // if(axis == Axis::COL){
    //   if(keepdim) res = Tensor<T>(row, 1, channel, 0, number);
    //   else        res = Tensor<T>(1, row, channel, 0, number);
    //   for(int n = 0; n < number; n++){
    //     int noffset = volume * n, roffset = row * channel * n;
    //     if(channel >= ncpu * BOOST_CHANNEL / number){
    //       parallel_channel(operator_axis0_channel<T>,
    //       /*nthread, res */NTHREAD_C(ncpu, number), res,
    //       /*const args...*/*this, noffset, roffset, mode);
    //     }
    //     else if(row >= ncpu * BOOST_ROW){
    //       parallel_row    (operator_axis0_row<T>, 
    //       /*nthread, res */NTHREAD_R(ncpu, number), res,
    //       /*const args...*/*this, noffset, roffset, mode);
    //     }
    //     else{ // Not need to boost.
    //       operator_axis0_channel(0, channel, 0, res,
    //                              *this, noffset, roffset, mode); 
    //     }
    //   }
    // } // axix == col
    // else if(axis == Axis::ROW){
    //   res = Tensor<T>(1, col, channel, 0, number);
    //   for(int n = 0; n < number; n++){
    //     int noffset = volume * n, roffset = col * channel * n;
    //     if(channel >= ncpu * BOOST_CHANNEL){
    //       parallel_channel(operator_axis1_channel<T>, 
    //       /*nthread, res */NTHREAD_C(ncpu, number), res,
    //       /*const args...*/*this, noffset, roffset, mode);
    //     }
    //     else if(row >= ncpu * BOOST_ROW){
    //       parallel_col    (operator_axis1_col<T>, 
    //       /*nthread, res */NTHREAD_R(ncpu, number), res,
    //       /*const args...*/*this, noffset, roffset, mode);
    //     }
    //     else{ // Not need to boost.
    //       int start = noffset, end = start + volume;
    //       operator_axis1_channel(0, channel, 0, res,
    //                              *this, noffset, roffset, mode); 
    //     }
    //   }
    // } // axix == row
    // else if(axis == Axis::CHANNEL){
    //   res = Tensor<T>(row, col, 1, 0, number);
    //   for(int n = 0; n < number; n++){
    //     int noffset = volume * n, roffset = row * col * n;
    //     if(row >= ncpu * BOOST_ROW){
    //       parallel_row    (operator_axis2_row<T>, 
    //       /*nthread, res */NTHREAD_C(ncpu, number), res,
    //       /*const args...*/*this, noffset, roffset, mode);
    //     }
    //     else if(col >= ncpu * BOOST_ROW){
    //       parallel_col    (operator_axis2_col<T>, 
    //       /*nthread, res */NTHREAD_R(ncpu, number), res,
    //       /*const args...*/*this, noffset, roffset, mode);
    //     }
    //     else{ // Not need to boost.
    //       operator_axis2_row(0, row, channel, res,
    //                          *this, noffset, roffset, mode); 
    //     }
    //   }
    // } // axis == channel
  }


  template<typename U>
  std::ostream &
  operator<<(std::ostream& os, const Tensor<U>& t){
    int row = t.row(), col = t.col(), channel = t.channel(), number = t.number();
    int square = row * col, volume = square * channel;

    // os.setf(std::ios::scientific);  // 科学计数法
    os.precision(PRINT_PRECISION);

    for(int n = 0; n < number; n++){
      int noffset = volume * n;
      if(row == 1 && channel == 1){
        printf("[");
        for(int i = 0; i < col; i++){
          os << std::setw(4) << t[noffset + i]; if(i != col - 1) os << ",";
        }
        printf("]\n");
      }
      if(row > 1 && channel == 1){
        printf("[");
        for(int r = 0; r < row; r++){
          int row_idx = noffset + col * r;
          if(r != 0)             printf(" ");
          printf("[");
          for(int c = 0; c < col; c++){
            os << std::setw(4) << t[row_idx + c]; if(c != col - 1) os << ",";
          }
          printf("]");
          if(r != row - 1)       printf("\n");
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
              os << std::setw(4) << t[row_idx + c]; if(c != col - 1) os << ",";
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

