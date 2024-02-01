#pragma once

#include "conv.hh"

namespace dl{

template<typename T=f32>
class ResidualBlock : public Function<T>{
public:
  explicit
  ResidualBlock(int input_ch, int output_ch) {
    same_shape = (input_ch == output_ch);
    input_layer = new Conv2D(1, input_ch, output_ch);
    output_layer = new Conv2D(1, output_ch, output_ch);
    if(!same_shape){
      pad_layer = new Conv2D(1, input_ch, output_ch);
    }
  }

  ~ResidualBlock(){
    delete input_layer;
    delete output_layer;
    delete pad_layer;
  }

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
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

private:
  Conv2D<T> *input_layer;
  Conv2D<T> *output_layer;
  Conv2D<T> *pad_layer;
  bool same_shape;
};


template<typename T=f32>
class ResidualBlock_bottle : public Function<T>{
public:
  explicit
  ResidualBlock_bottle(int input_ch, int neck_ch, int output_ch) {
    same_shape = (input_ch == output_ch);
    input_layer = new Conv2D(1, input_ch, neck_ch);
    neck_layer = new Conv2D(3, neck_ch, neck_ch, 1, 1);
    output_layer = new Conv2D(1, neck_ch, output_ch);
    if(!same_shape){
      pad_layer = new Conv2D(1, input_ch, output_ch);
    }
  }

  virtual ~ResidualBlock_bottle(){
    puts("invoke ~Residual dtor");
    delete input_layer;
    delete neck_layer;
    delete output_layer;
    if(same_shape){
      delete pad_layer;
    }
  }

  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<Tensor<T>> input){
    auto output1 = input_layer->forward(input);
    auto output2 = neck_layer->forward(output1);
    auto output3 = output_layer->forward(output2);
    if(same_shape){
      auto output = *output3 + *input;
      return output;
    }
    else{
      auto pad_input = pad_layer->forward(input);
      auto output = *output3 + *pad_input;
      return output;
    }
  }

private:
  Conv2D<T> *input_layer;
  Conv2D<T> *neck_layer;
  Conv2D<T> *output_layer;
  Conv2D<T> *pad_layer;
  bool same_shape;
};
}