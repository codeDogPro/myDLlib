#pragma once

#include <cuda_device_runtime_api.h>
#include <data/rand_init.cuh>
#include <parallel/parallel.cuh>
#include <parallel/tensor_cpu.cuh>
#include <parallel/tensor_cuda.cuh>
#include <parallel/basic_cuda.cuh>

#include <numeric>
#include <cstring>
#include <iomanip>


namespace dl{

template<typename T=f32>
class Tensor{

public:
  explicit Tensor() = default;

  ~Tensor() = default;

  explicit
  Tensor(int row, int col, int ch=1, int num=1, T val=T(-1),
         Device device=Device::CPU)
  {
    assert(row != 0 && col != 0 && ch != 0 && num != 0);

    if(device == Device::CPU){
      m_hostData.assign(row * col * ch * num, val);
      if(val == T(-1)){ rand_init_cpu<T>(m_hostData);}
    }
    else if(device == Device::CUDA){
      m_cudaData.assign(row * col * ch * num, val);
      if(val == T(-1)) rand_init_cuda<T>(m_cudaData);
    }
    m_shape = {row, col, ch, num};
    full_print = false;
    m_device = device;
  }

  explicit
  Tensor(const std::vector<int>& shape, T val=-1, Device device=Device::CPU)
  { assert(shape.size() != 0);

    int size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<T>{});
    if(device == Device::CPU) {
      m_hostData.assign(size, val);
      if(val == T(-1)) rand_init_cpu<T>(m_hostData);
    }
    else if(device == Device::CUDA) {
      m_cudaData.assign(size, val);
      if(val == T(-1)) rand_init_cuda<T>(m_cudaData);
    }
    m_shape = shape;
    m_device = device;
    full_print = false;
  }

  explicit
  Tensor(thrust::host_vector<T, AlignedAllocator<T, 64>>& data, std::vector<int>& shape)
  : m_hostData(data), m_shape(shape), full_print(false)
  {
    m_device = Device::CPU;
  }

  // deep copy
  explicit
  Tensor(const Tensor<T>& t){ 
    m_shape.assign(4, 0); m_hostData.assign(t.size(), 0);
    for(int i = 0; int x : t.get_cshape()) m_shape[i++] = x;
    parallel_copy(t);
    full_print = false;
    m_device = Device::CPU;
  }

  // move copy ctor
  Tensor(Tensor<T>&& t){ 
    if(t.device() == Device::CPU){
      m_hostData  = t.get_data();
      m_shape = t.get_shape();
      full_print = false;
      m_device = Device::CPU;
    }
  }

  Tensor<T> &
  operator=(const Tensor<T>& rhs){
    if(this == &rhs) return *this;

    if(rhs.device() == Device::CPU){
      // puts("invoke operator= copy");
      m_shape.assign(4, 0); m_hostData.assign(rhs.size(), 0);
      for(int i = 0; int x : rhs.get_cshape()) m_shape[i++] = x;

      parallel_copy(rhs);
      // puts("Finish operator= copy");
      full_print = false;
      m_device = Device::CPU;
    }
    return *this;
  }

  Tensor<T> &
  operator=(Tensor<T>&& rhs){
    if(this == &rhs) return *this;

    if(rhs.device() == Device::CPU){
      // puts("invoke operator= move");
      m_hostData  = std::move(rhs.get_data());
      m_shape = std::move(rhs.get_shape());
      full_print = false;
      m_device = Device::CPU;
    }
    return *this;
  }

  void to(Device device){
    if(device == m_device) return ;

    if(device == Device::CPU){
      m_hostData = m_cudaData;
      m_device = Device::CPU;
      printf("cudaData size: %ld\n", m_cudaData.size());
    }
    else if(device == Device::CUDA){
      m_cudaData = m_hostData;
      m_device = Device::CUDA;
      printf("hostData size: %ld\n", m_hostData.size());
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
  std::vector<int> &      get_shape ()       { return m_shape;}
  std::vector<int> const& get_cshape() const { return m_shape;}
  thrust::host_vector<T, AlignedAllocator<T, 64>> &      
  get_data()      {return m_hostData;}
  thrust::host_vector<T, AlignedAllocator<T, 64>> const& 
  get_cdata()const{return m_hostData;}

  // for gpu kernel to call
  thrust::device_ptr<T>       data_gpu()       { return m_cudaData.data(); } 
  thrust::device_ptr<const T> data_gpu() const { return m_cudaData.data(); } 
  thrust::device_vector<T>        & get_data_gpu()         { return m_cudaData;}
  thrust::device_vector<T>   const& get_cdata_gpu()  const { return m_cudaData;}

  bool reshape(const std::vector<int>& shape){ 
    size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<T>{});
    if(size != m_hostData.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_hostData.size());
      exit(-1);
    }
    m_shape = shape;
    return true;
  } 

  bool reshape(int row, int col, int channel, int number=1){
    size_t size = row * col * channel * number;
    if(size != m_hostData.size()){
      fprintf(stderr, "New size:%ld isn't equal to the data size:%ld\n",
      size, m_hostData.size());
      exit(-1);
    }
    m_shape[0] = row, m_shape[1] = col, m_shape[2] = channel, m_shape[3] = number;
    return true;
  } 

  int size() {
    return std::reduce(m_shape.begin(), m_shape.end(),
                       1, std::multiplies{});
  }
  int size() const {
    return std::reduce(m_shape.begin(), m_shape.end(),
                       1, std::multiplies{});
  }

  // for cpu kernel to call [most common]
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
  
  // only print in cpu
  void setFullPrintMode(bool mode)       { full_print = mode; }
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
  // [0]:row [1]:col [2]:channel [3]:number
  /*shape is only stored in cpu memory,
    so we should give the tensor shape as 
    parameter to cuda kernel instead just 
    access the shape in kernel; */
  std::vector<int> m_shape;

  thrust::host_vector<T, AlignedAllocator<T, 64>> m_hostData;
  thrust::device_vector<T> m_cudaData;

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
    Tensor<T> rhs(this->m_shape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this + rhs; 
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator-(T x) { 
    Tensor<T> rhs(this->m_shape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this - rhs; 
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator*(T x) { 
    Tensor<T> rhs(this->m_shape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this * rhs; 
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> 
  Tensor<T>::operator/(T x) { 
    Tensor<T> rhs(this->m_shape, x); 
    if(m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    return *this / rhs; 
  }

  template<typename T>
  void 
  Tensor<T>::operator+=(T x) {
    Tensor<T> rhs(this->m_shape, x);
    if(this->m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    *this += rhs;
  }

  template<typename T>
  void 
  Tensor<T>::operator-=(T x) {
    Tensor<T> rhs(this->m_shape, x);
    if(this->m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    *this -= rhs;
  }

  template<typename T>
  void 
  Tensor<T>::operator*=(T x) {
    Tensor<T> rhs(this->m_shape, x);
    if(this->m_device == Device::CUDA){
      rhs.to(Device::CUDA);
    }
    *this *= rhs;
  }

  template<typename T>
  void 
  Tensor<T>::operator/=(T x) {
    Tensor<T> rhs(this->m_shape, x);
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
    int lrow = lhs.row(), rrow = rhs.row(), col = lhs.col();
    int channel = lhs.channel(), number = lhs.number();
    int orow = std::max(lrow, rrow);

    auto output = std::make_shared<Tensor<T>>(orow, col, channel, number, 0);
    int volume = orow * col * channel;
    const T *_lhs, *_rhs;
    if(lrow == rrow){
      _lhs = lhs.data(), _rhs = rhs.data();

      if(channel >= number * BOOST_CHANNEL){
        for(int i = 0; i < number; i++){
          int offset = i * volume;
          switch(mode){
            case Calculator::ADD:
              parallelizer.parallel_channel(
                vec_add_full<T>, output, offset, _lhs, _rhs); break;
            case Calculator::SUB:
              parallelizer.parallel_channel(
                vec_sub_full<T>, output, offset, _lhs, _rhs); break;
            case Calculator::MUL:
              parallelizer.parallel_channel(
                vec_mul_full<T>, output, offset, _lhs, _rhs); break;
            case Calculator::DIV:
              parallelizer.parallel_channel(
                vec_div_full<T>, output, offset, _lhs, _rhs); break;
            default: assert(0); 
          }
        }
      }
      else {
        switch(mode){
          case Calculator::ADD:
            parallelizer.parallel_number(
              vec_add_full<T>, output, _lhs, _rhs); break;
          case Calculator::SUB:
            parallelizer.parallel_number(
              vec_sub_full<T>, output, _lhs, _rhs); break;
          case Calculator::MUL:
            parallelizer.parallel_number(
              vec_mul_full<T>, output, _lhs, _rhs); break;
          case Calculator::DIV:
            parallelizer.parallel_number(
              vec_div_full<T>, output, _lhs, _rhs); break;
          default: assert(0); 
        }
      }
    }
    else{ // lhs.row != rhs.row
      if(lrow < rrow){
        if(lrow == 1)
          _lhs = rhs.data(), _rhs = lhs.data();
        else goto erro;
      }
      else{ // lrow > rrow
        if(rrow == 1)
          _lhs = lhs.data(), _rhs = rhs.data();
        else goto erro;
      }

      if(channel >= number * BOOST_CHANNEL){
        for(int i = 0; i < number; i++){
          int offset = i * volume;
          switch(mode){
            case Calculator::ADD:
              parallelizer.parallel_channel(
                vec_add_single<T>, output, offset, _lhs, _rhs); break;
            case Calculator::SUB:
              parallelizer.parallel_channel(
                vec_sub_single<T>, output, offset, _lhs, _rhs); break;
            case Calculator::MUL:
              parallelizer.parallel_channel(
                vec_mul_single<T>, output, offset, _lhs, _rhs); break;
            case Calculator::DIV:
              parallelizer.parallel_channel(
                vec_div_single<T>, output, offset, _lhs, _rhs); break;
            default: assert(0); 
          }
        }
      }
      else{
        switch(mode){
          case Calculator::ADD:
            parallelizer.parallel_number(
              vec_add_single<T>, output, _lhs, _rhs); break;
          case Calculator::SUB:
            parallelizer.parallel_number(
              vec_sub_single<T>, output, _lhs, _rhs); break;
          case Calculator::MUL:
            parallelizer.parallel_number(
              vec_mul_single<T>, output, _lhs, _rhs); break;
          case Calculator::DIV:
            parallelizer.parallel_number(
              vec_div_single<T>, output, _lhs, _rhs); break;
          default: assert(0); 
        }
      }
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

    auto output = std::make_shared<Tensor<T>>
      (orow, col, channel, number, 0, Device::CUDA);
    int size = output->size();
    int block_size = 128, grid_size = 64;

    thrust::device_ptr<const T> _lhs, _rhs;
    thrust::device_ptr<T> _output = output->data_gpu();
    if(lrow == rrow){
      _lhs = lhs.data_gpu(), _rhs = rhs.data_gpu();
      switch(mode){
        case Calculator::ADD:
          cuda_add_full<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size); break;
        case Calculator::SUB:
          cuda_sub_full<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size); break;
        case Calculator::MUL:
          cuda_mul_full<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size); break;
        case Calculator::DIV:
          cuda_div_full<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size); break;
        default: exit(-1);
      } 
    }
    else{
      if(lrow < rrow){
        if(lrow == 1)
          _lhs = rhs.data_gpu(), _rhs = lhs.data_gpu();
        else goto erro;
      }
      else {
        if(rrow == 1)
          _lhs = lhs.data_gpu(), _rhs = rhs.data_gpu();
        else goto erro;
      }
      switch(mode){
        case Calculator::ADD:
          cuda_add_single<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size, orow, col); break;
        case Calculator::SUB:
          cuda_sub_single<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size, orow, col); break;
        case Calculator::MUL:
          cuda_mul_single<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size, orow, col); break;
        case Calculator::DIV:
          cuda_div_single<T><<<grid_size, block_size>>>
            (_lhs, _rhs, _output, size, orow, col); break;
        default: exit(-1);
      } 
    }
    cudaDeviceSynchronize();
    return output;

  erro:
    fprintf(stderr, "The size of tensor lhs:(%d) must match the size of \
    tensor rhs:(%d) at non-singleton dimension 0\n", lhs.row(), rhs.row());
    exit(-1);
  }
  
  template<typename T>
  std::shared_ptr<Tensor<T>>
  Tensor<T>::tensor_operator_cuda(Axis axis, Operator mode, bool keepdim){
    int row = this->row(), col = this->col(), ch = this->channel();
    int num = this->number(), size = this->size();
    std::shared_ptr<Tensor<T>> output;
    if(axis == Axis::COL){
      if(keepdim == true){
        output = std::make_shared<Tensor<T>>(row, 1, ch, num, 0, Device::CUDA);
      }
      else{
        output = std::make_shared<Tensor<T>>(1, row, ch, num, 0, Device::CUDA);
      }
      auto _input = this->data_gpu(), _output = output->data_gpu();
      constexpr int tile_x = 32, tile_y = 32;
      int grid_y = (size / col + tile_y - 1) / tile_y;
      int grid_x = (col + tile_x - 1) / tile_x;
      // printf("grid_y: %d grid_x:%d\n", grid_y, grid_x);
      dim3 grid_size(grid_x, grid_y), block_size(tile_x, tile_y);
      switch(mode){
        case Operator::SUM:
          reduce4D_axis0_cuda<<<grid_size, block_size>>>
            (_input, _output, size, col);
          break;
        case Operator::MEAN:
          reduce4D_axis0_cuda<<<grid_size, block_size>>>
            (_input, _output, size, col); 
          cudaDeviceSynchronize();
          mean_axis0_cuda<<<64, 128>>>
            (_output, _output, size, col);
          break;
        case Operator::MAX:
        case Operator::MIN:
        default: exit(-1);
      }
    }
    else if(axis == Axis::ROW){
    }
    else if(axis == Axis::CHANNEL){
    }
    cudaDeviceSynchronize();
    return output;
  }


//##################################################################################################
//##################################################################################################
//######################################### PRINT ##################################################
  template<typename T>
  void print_H(std::ostream &os, const Tensor<T> &t, int offset){
    using std::setw;
    // os.setf(std::ios::scientific);  // 科学计数法
    os.precision(PRINT_PRECISION);
    int channel = t.channel(), number = t.number();

    int col = t.col();
    if(col > MAX_PRINT_COL){
      if(t.getFullPrintMode() == true){
        int num_cnt = 0;
        for(int i = 0; i < col; i++){
          os << setw(7) << t[offset + i];
          if(i != col - 1) os << ", ";
          if(++num_cnt == MAX_NUM_LINE){
            if(number > 1)    os << "\n    ";
            else {
              if(channel > 1) os << "\n   ";
              else            os << "\n  ";
            }
            num_cnt = 0;
          }
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
      int num_cnt = 0;
      for(int i = 0; i < col; i++){
        os << setw(7) << t[offset + i];
        if(i != col - 1) os << ", ";
        if(++num_cnt == MAX_NUM_LINE){
          if(number > 1)    os << "\n    ";
          else {
            if(channel > 1) os << "\n   ";
            else            os << "\n  ";
          }
          num_cnt = 0;
        }
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
          if(number > 1){
            if(r != 0) printf("   "); 
          }
          else{
            if(r != 0) printf("  "); 
          }
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
        if(number > 1){
          if(r != 0) printf("   "); 
        }
        else{
          if(r != 0) printf("  "); 
        }
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
//########################################### END PRINT ###################################################
//##################################################################################################
//##################################################################################################



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

