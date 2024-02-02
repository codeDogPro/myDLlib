#pragma once

#include <data/rand_init.cuh>
#include <data/align_alloc.cuh>

#include <parallel/parallel.cuh>
#include <parallel/tensor_cpu.cuh>
#include <parallel/tensor_cuda.cuh>

#include <numeric>
#include <cstring>
#include <iomanip>

#include <opencv2/core.hpp>

// cuda accleration
#include <thrust/host_vector.h>

namespace dl{

template<typename T=f32>
class Tensor{

public:
  explicit Tensor() = default;

  ~Tensor() = default;

  explicit
  Tensor(int row, int col, int ch=1, int num=1, T val=T(-1),
   Device device=Device::CPU){
    assert(row != 0 && col != 0 && cha != 0 && num != 0);

    if(device == Device::CPU){
      m_hostData.assign(row * col * ch * num, val);
    }
    else if(device == Device::CUDA){
      m_cudaData.assign(row * col * ch * num, val);
    }
    m_hostShape.assign(4, 0);
    m_hostShape[0] = row, m_hostShape[1] = col;
    m_hostShape[2] = ch, m_hostShape[3] = num; 
    m_cudaShape = m_hostShape;

    full_print = false;
    m_device = device;
    if(val == T(-1)){ rand_init(*this);}
  }

  explicit
  Tensor(const thrust::host_vector<int>& shape, T val=-1){
    assert(shape.size() != 0);

    int size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    m_hostData.assign(size, val);
    m_hostShape = shape;
    full_print = false;
    m_device = Device::CPU;
    if(val == T(-1)){ rand_init(*this);}
  }

  explicit
  Tensor(thrust::host_vector<T, AlignedAllocator<T, 64>>& data, thrust::host_vector<int>& shape)
  : m_hostData(data), m_hostShape(shape), full_print(false) {}

  // deep copy
  explicit
  Tensor(const Tensor<T>& t){ 
    m_hostShape.assign(4, 0); m_hostData.assign(t.size(), 0);
    int i = 0; 
    for(int x : t.get_cshape()) m_hostShape[i++] = x;
    parallel_copy(t);
    full_print = false;
    m_device = Device::CPU;
  }

  // move copy ctor
  Tensor(Tensor<T>&& t){ 
    m_hostData  = t.get_data();
    m_hostShape = t.get_shape();
    full_print = false;
    m_device = Device::CPU;
  }

  Tensor<T> &
  operator=(const Tensor<T>& rhs){
    if(this == &rhs) return *this;

    // puts("invoke operator= copy");
    m_hostShape.assign(4, 0); m_hostData.assign(rhs.size(), 0);
    int i = 0; 
    for(int x : rhs.get_cshape()) m_hostShape[i++] = x;

    parallel_copy(rhs);
    // puts("Finish operator= copy");
    full_print = false;
    m_device = Device::CPU;
    return *this;
  }

  Tensor<T> &
  operator=(Tensor<T>&& rhs){
    if(this == &rhs) return *this;

    puts("invoke operator= move");
    m_hostData  = std::move(rhs.get_data());
    m_hostShape = std::move(rhs.get_shape());
    full_print = false;
    m_device = Device::CPU;
    return *this;
  }

  void to(Device device){
    if(device == m_device) return ;

    if(device == Device::CPU){
      m_hostData = m_cudaData;
      m_hostShape = m_cudaShape;
      m_device = Device::CPU;
    }
    else if(device == Device::CUDA){
      m_cudaData = m_hostData;
      m_cudaShape = m_hostShape;
      m_device = Device::CUDA;
    }
    else {
      fprintf(stderr, "only support cpu and cuda\n");
      exit(-1);
    }
  }


  std::shared_ptr<Tensor<T>> operator+(const Tensor<T>& rhs);
  std::shared_ptr<Tensor<T>> operator-(const Tensor<T>& rhs);
  std::shared_ptr<Tensor<T>> operator*(const Tensor<T>& rhs);
  std::shared_ptr<Tensor<T>> operator/(const Tensor<T>& rhs);
  std::shared_ptr<Tensor<T>> operator+(T x);
  std::shared_ptr<Tensor<T>> operator-(T x);
  std::shared_ptr<Tensor<T>> operator*(T x);
  std::shared_ptr<Tensor<T>> operator/(T x);

  void operator+=(T x);
  void operator-=(T x);
  void operator*=(T x);
  void operator/=(T x);
  void operator+=(const Tensor<T>& rhs) { *this = *(*this + rhs); }
  void operator-=(const Tensor<T>& rhs) { *this = *(*this - rhs); }
  void operator*=(const Tensor<T>& rhs) { *this = *(*this * rhs); }
  void operator/=(const Tensor<T>& rhs) { *this = *(*this / rhs); }

  // for cpu func to access Tensor's data
        T& operator[](size_t idx)       { return m_hostData[idx];}
  const T& operator[](size_t idx) const { return m_hostData[idx];}

  // for gpu kernel to access Tensor's data
        T& at(size_t idx)       { return m_cudaData[idx];}
  const T& at(size_t idx) const { return m_cudaData[idx];}

  template<typename U>
  friend std::ostream & operator<<(std::ostream &os, Tensor<U> &t);

  std::shared_ptr<Tensor<T>> sum (int axis=0, bool keepdim=false);
  std::shared_ptr<Tensor<T>> mean(int axis=0, bool keepdim=false);
  std::shared_ptr<Tensor<T>> max (int axis=0, bool keepdim=false);
  std::shared_ptr<Tensor<T>> min (int axis=0, bool keepdim=false);

  // for cpu to call
        T *data()       { return m_hostData.data(); }
  const T *data() const { return m_hostData.data(); }
  thrust::host_vector<int> &      get_shape ()       { return m_hostShape;}
  thrust::host_vector<int> const& get_cshape() const { return m_hostShape;}
  thrust::host_vector<T, AlignedAllocator<T, 64>> &      get_data()      {return m_hostData;}
  thrust::host_vector<T, AlignedAllocator<T, 64>> const& get_cdata()const{return m_hostData;}

  // for gpu kernel to call
  thrust::device_ptr<T>       data_gpu()       { return m_cudaData.data(); } 
  thrust::device_ptr<const T> data_gpu() const { return m_cudaData.data(); } 
  thrust::device_vector<int>      & get_shape_gpu ()       { return m_cudaShape;}
  thrust::device_vector<int> const& get_cshape_gpu() const { return m_cudaShape;}
  thrust::device_vector<T>        & get_data_gpu()         { return m_cudaData;}
  thrust::device_vector<T>   const& get_cdata_gpu()  const { return m_cudaData;}

  bool reshape(const thrust::host_vector<int>& shape){ 
    size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});
    if(size != m_hostData.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_hostData.size());
      exit(-1);
    }
    m_hostShape = shape;
    // just change the shape
    if(m_device == Device::CUDA) {
      m_cudaShape = m_hostShape;
    }
    return true;
  } 

  bool reshape(int row, int col, int channel, int number=1){
    size_t size = row * col * channel * number;
    if(size != m_hostData.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_hostData.size());
      exit(-1);
    }
    m_hostShape[0] = row, m_hostShape[1] = col, m_hostShape[2] = channel, m_hostShape[3] = number;
    // just change the shape
    if(m_device == Device::CUDA) {
      m_cudaShape = m_hostShape;
    }
    return true;
  } 

  size_t size() {
    return std::reduce(m_hostShape.begin(), m_hostShape.end(), 1, std::multiplies{});
  }
  size_t size() const {
    return std::reduce(m_hostShape.begin(), m_hostShape.end(), 1, std::multiplies{});
  }

  // for cpu kernel to call [most common]
  int row()           { return m_hostShape[0]; }
  int col()           { return m_hostShape[1]; }
  int channel()       { return m_hostShape[2]; }  
  int number()        { return m_hostShape[3]; }
  int row()     const { return m_hostShape[0]; }
  int col()     const { return m_hostShape[1]; }
  int channel() const { return m_hostShape[2]; }  
  int number()  const { return m_hostShape[3]; }

  void shape(){
    printf("shape:[");
    for(int i = 0; i < m_hostShape.size(); i++) {
      std::cout << m_hostShape[i];
      if(i != m_hostShape.size() - 1) printf(", ");
    }
    printf("]\n");
  }
  
  // only print in cpu
  void setFullPrintMode(bool mode)       { full_print = mode; }
  void setFullPrintMode(bool mode) const { full_print = mode; }
  bool getFullPrintMode()       { return full_print; }
  bool getFullPrintMode() const { return full_print; }
  Device device()       { return m_device; }
  Device device() const { return m_device; }

protected:
  void parallel_copy(const Tensor<T> &rhs);
  std::shared_ptr<Tensor<T>> 
  tensor_calculator(const Tensor<T>& lhs, const Tensor<T>& rhs, Calculator mode);
  std::shared_ptr<Tensor<T>> 
  tensor_operator(Axis axis, Operator mode, bool keepdim);
  std::shared_ptr<Tensor<T>> 
  tensor_calculator_cuda(const Tensor<T>& lhs, const Tensor<T>& rhs, Calculator mode);
  std::shared_ptr<Tensor<T>> 
  tensor_operator_cuda(Axis axis, Operator mode, bool keepdim);

private:
  thrust::host_vector<int> m_hostShape; // [0]:row [1]:col [2]:channel [3]:number
  thrust::host_vector<T, AlignedAllocator<T, 64>> m_hostData;

  thrust::device_vector<T> m_cudaData;
  thrust::device_vector<int> m_cudaShape;

  Device m_device;
  bool full_print;
}; // class Tensor


//################### Tensor::member functions' implementation ###################

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator+(const Tensor<T>& rhs) { 
    if(this->m_device == Device::CPU)
      return tensor_calculator(*this, rhs, Calculator::ADD);
    else
      return tensor_calculator_cuda(*this, rhs, Calculator::ADD);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator-(const Tensor<T>& rhs) { 
    if(this->m_device == Device::CPU)
      return tensor_calculator(*this, rhs, Calculator::SUB);
    else
      return tensor_calculator_cuda(*this, rhs, Calculator::SUB);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator*(const Tensor<T>& rhs) { 
    if(this->m_device == Device::CPU)
      return tensor_calculator(*this, rhs, Calculator::MUL);
    else
      return tensor_calculator_cuda(*this, rhs, Calculator::MUL);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator/(const Tensor<T>& rhs) { 
    if(this->m_device == Device::CPU)
      return tensor_calculator(*this, rhs, Calculator::DIV);
    else
      return tensor_calculator_cuda(*this, rhs, Calculator::DIV);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator+(T x) { 
    Tensor<T> rhs(this->m_hostShape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this + rhs; 
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator-(T x) { 
    Tensor<T> rhs(this->m_hostShape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this - rhs; 
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator*(T x) { 
    Tensor<T> rhs(this->m_hostShape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this * rhs; 
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator/(T x) { 
    Tensor<T> rhs(this->m_hostShape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this / rhs; 
  }

  template<typename T>
  void 
  Tensor<T>::operator+=(T x) {
    Tensor<T> rhs(this->m_hostShape, x);
    if(this->m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    *this += rhs;
  }

  template<typename T>
  void 
  Tensor<T>::operator-=(T x) {
    Tensor<T> rhs(this->m_hostShape, x);
    if(this->m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    *this -= rhs;
  }

  template<typename T>
  void 
  Tensor<T>::operator*=(T x) {
    Tensor<T> rhs(this->m_hostShape, x);
    if(this->m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    *this *= rhs;
  }

  template<typename T>
  void 
  Tensor<T>::operator/=(T x) {
    Tensor<T> rhs(this->m_hostShape, x);
    if(this->m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    *this /= rhs;
  }

  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::sum(int axis, bool keepdim){ 
    return tensor_operator(Axis(axis), Operator::SUM, keepdim);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::mean(int axis, bool keepdim){
    return tensor_operator(Axis(axis), Operator::MEAN, keepdim);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::max(int axis, bool keepdim){
    return tensor_operator(Axis(axis), Operator::MAX, keepdim);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::min(int axis, bool keepdim){ 
    return tensor_operator(Axis(axis), Operator::MIN, keepdim);
  }

  template<typename T>
  void 
  Tensor<T>::parallel_copy(const Tensor<T> &rhs){
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
  
  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::tensor_calculator_cuda
  (const Tensor<T>& lhs, const Tensor<T>& rhs, Calculator mode){
    assert(lhs.col() == rhs.col() && lhs.channel() == rhs.channel() &&
           lhs.number() == rhs.number());
    int lrow = lhs.row(), col = lhs.col(), rrow = rhs.row();
    int channel = lhs.channel(), number = lhs.number();
    int orow = std::max(lrow, rrow);

    auto output = std::make_shared<Tensor<T>>(
      orow, col, channel, number, 0, Device::CUDA);
    int size = output->size();
    int block_size = 128, grid_size = size / block_size;

    if(lrow == rrow){
      switch(mode){
        case Calculator::ADD:
          cuda_add_full<<<grid_size, block_size>>>
            (lhs.data_gpu(), rhs.data_gpu(), output->data_gpu(), size); break;
        case Calculator::SUB:
          cuda_sub_full<<<grid_size, block_size>>>
            (lhs.data_gpu(), rhs.data_gpu(), output->data_gpu(), size); break;
        case Calculator::MUL:
          cuda_mul_full<<<grid_size, block_size>>>
            (lhs.data_gpu(), rhs.data_gpu(), output->data_gpu(), size); break;
        case Calculator::DIV:
          cuda_div_full<<<grid_size, block_size>>>
            (lhs.data_gpu(), rhs.data_gpu(), output->data_gpu(), size); break;
        default: exit(-1);
      } 
    }
    else{
      // switch(mode){
      //   case Calculator::ADD:
      //     cuda_add_single<<<1, 128>>>
      //       (lhs.data(), rhs.data(), output->data(), lhs.get_cshape_gpu()); break;
      //   case Calculator::SUB:
      //     cuda_sub_single<<<1, 128>>>
      //       (lhs.data(), rhs.data(), output->data(), lhs.get_cshape_gpu()); break;
      //   case Calculator::MUL:
      //     cuda_mul_single<<<1, 128>>>
      //       (lhs.data(), rhs.data(), output->data(), lhs.get_cshape_gpu()); break;
      //   case Calculator::DIV:
      //     cuda_div_single<<<1, 128>>>
      //       (lhs.data(), rhs.data(), output->data(), lhs.get_cshape_gpu()); break;
      //   default: exit(-1);
      // } 
    }
    return output;
  }
  
  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::tensor_operator_cuda(Axis axis, Operator mode, bool keepdim){
    // TODO: implement it
    return nullptr;
  }

//######################################### PRINT ##################################################
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
  operator<<(std::ostream& os, Tensor<U>& t){
    // insure the Tensor's data are in CPU memory
    if(t.device() == Device::CUDA){
      t.to(Device::CPU);
    }
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

