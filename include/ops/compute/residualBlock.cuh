#pragma once

#include <basic/tensor_macro.cuh>
#include "conv.cuh"

namespace dl{

template<typename T=f32>
class ResidualBlock : public Function<T>{
public:
  explicit
  ResidualBlock(int input_ch, int output_ch, Device device=Device::CPU) {
    same_shape = (input_ch == output_ch);
    input_layer = new Conv2D<T>(1, input_ch, output_ch, 1, 0, device);
    output_layer = new Conv2D<T>(1, output_ch, output_ch, 1, 0, device);
    if(!same_shape){
      pad_layer = new Conv2D<T>(1, input_ch, output_ch, 1, 0, device);
    }
    m_device  = device;
  }

  virtual ~ResidualBlock(){
    delete input_layer;
    delete output_layer;
    if(!same_shape){
      delete pad_layer;
    }
  }

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<const T>> input){
    auto output1 = input_layer->forward(input);
    auto output2 = output_layer->forward(output1);
    if(same_shape){
      auto output = *output2 + *input;
      return output;
    }
    else{
      auto pad_input = pad_layer->forward(input);
      auto output = *output2 + *pad_input;
      return output;
    }
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<Tensor<const T>> input){
    return forward(input);
  }

  void to(Device device){
    input_layer->to(device);
    output_layer->to(device);
    pad_layer->to(device);
    m_device = device;
  }

private:
  Conv2D<T> *input_layer;
  Conv2D<T> *output_layer;
  Conv2D<T> *pad_layer;
  bool same_shape;
  Device m_device;
};


template<typename T=f32>
class ResidualBlock_bottle : public Function<T>{
public:
  explicit
  ResidualBlock_bottle(int input_ch, int neck_ch, int output_ch, Device device=Device::CPU) {
    same_shape = (input_ch == output_ch);
    input_layer = new Conv2D<T>(1, input_ch, neck_ch, 1, 0, device);
    neck_layer = new Conv2D<T>(3, neck_ch, neck_ch, 1, 1, device);
    output_layer = new Conv2D<T>(1, neck_ch, output_ch, 1, 0, device);
    if(!same_shape){
      pad_layer = new Conv2D<T>(1, input_ch, output_ch, 1, 0, device);
      //* 手动同步
      pad_layer->setSyncMode(false);
    }
  }

  virtual ~ResidualBlock_bottle(){
    delete input_layer;
    delete neck_layer;
    delete output_layer;
    if(!same_shape){
      delete pad_layer;
    }
  }

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<const Tensor<T>> input){
    // std::cout << "output3:\t"; output3->shape();
    if(same_shape == true){
      auto output1 = (*input_layer)(input);
      // std::cout << "output1:\t"; output1->shape();
      auto output2 = (*neck_layer)(output1);
      // std::cout << "output2:\t"; output2->shape();
      auto output3 = (*output_layer)(output2);
      auto output = *output3 + *input;
      // std::cout << "output:\t"; output->shape();
      return output;
    }
    else{
      auto pad_input = (*pad_layer)(input); //* unSync

      auto output1 = (*input_layer)(input);
      auto output2 = (*neck_layer)(output1);
      auto output3 = (*output_layer)(output2);

      auto output = *output3 + *pad_input;
      return output;
    }
  }

  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<const Tensor<T>> input){
    return forward(input);
  }

  void to(Device device){
    input_layer->to(device);
    neck_layer->to(device);
    output_layer->to(device);
    pad_layer->to(device);
    m_device = device;
  }

private:
  Conv2D<T> *input_layer;
  Conv2D<T> *neck_layer;
  Conv2D<T> *output_layer;
  Conv2D<T> *pad_layer;
  bool same_shape;
  Device m_device;
};

}