#pragma once

#include <basic/function.cuh>
#include <basic/tensor_macro.cuh>

#include <ops_impl/cuda/basic.cuh>
#include <ops_impl/cpu/activation.hh>
#include <ops_impl/cuda/activation.cuh>

namespace dl{

template<typename T=f32>
class Softmax : public Function<T> {
public:
  explicit Softmax() = default;

  explicit 
  Softmax(int axis=0) {
    M_axis = Axis(axis);
  };
  virtual ~Softmax() = default;

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<const Tensor<T>> input){
    return (input->device() == Device::CPU) ? forward_cpu(input) : forward_cuda(input);
  }

  std::shared_ptr<Tensor<T>> 
  operator()(const std::shared_ptr<const Tensor<T>> input){
    return forward(input);
  }

private:
  std::shared_ptr<Tensor<T>> 
  forward_cpu(const std::shared_ptr<const Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), Device::CPU, 0);
    const int row = input->row(), col = input->col(), channel = input->channel();
    const int number = input->number(), volume = row * col * channel;

    // get exp(input) first
    if(channel > 1){
      for(int i = 0; i < number; i++){
        const int offset = i * volume;
        parallelizer.parallel_channel(
          exp_cpu<T>, output, offset, input);
      }
    }
    else if(channel == 1 && row == 1){
      // in linear output
      for(int i = 0; i < number; i++){
        const int offset = i * volume;
        parallelizer.parallel_col(
          exp_cpu<T>, output, offset, input);
      }
    }
    parallelizer.sync();

    // then get softmax
    for(int i = 0; i < number; i++){
      const int offset = i * volume;
      if(M_axis == Axis::COL){
        parallelizer.parallel_channel(
          softmax_axis0_cpu<T>, output, offset, output->data());
      }
      else if(M_axis == Axis::ROW){
        parallelizer.parallel_channel(
          softmax_axis1_cpu<T>, output, offset, output->data());
      }
      else if(M_axis == Axis::CHANNEL){
        parallelizer.parallel_row(
          softmax_axis2_cpu<T>, output, offset, output->data());
      }
    }
    parallelizer.sync();
    return output;
  }

  std::shared_ptr<Tensor<T>> 
  forward_cuda(const std::shared_ptr<const Tensor<T>> input){
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), Device::CUDA, 0);
    int row = input->row(), col = input->col(), ch = input->channel();
    int num = input->number(), size = input->size();

    thrust::device_ptr<const T> _input = input->data_gpu();
    thrust::device_ptr<T>  _exp = output->data_gpu();;
    // calculate exp(input) first
    exp_cuda<T><<<64, 128>>>(_input, _exp, size);

    if(M_axis == Axis::COL){
      auto exp_sum = std::make_shared<Tensor<T>>(row, 1, ch, num, Device::CUDA, 0);
      auto _output = output->data_gpu(), _exp_sum = exp_sum->data_gpu();
      int grid_y = (size / col + TILE_Y - 1) / TILE_Y;
      int grid_x = (col + TILE_X - 1) / TILE_X;
      dim3 grid_size(grid_x, grid_y), block_size(TILE_X, TILE_Y);
      reduce4D_axis0_cuda<<<grid_size, block_size>>>
        (_exp, _exp_sum, size, col);
      softmax_axis0_cuda<T><<<64, 128>>>
        (_exp, _exp_sum, _output, size, col);
    }
    else if(M_axis == Axis::ROW){
      // TODO: not implemented
      // int grid_size = 64, block_size = 128;
      // softmax_axis1_cuda(_input, _output, size);
    }
    else if(M_axis == Axis::CHANNEL){
      // TODO: not implemented
      // int grid_size = 64, block_size = 128;
      // softmax_axis2_cuda(_input, _output, size);
    }
    return output;
  }

  Axis M_axis;
};
}