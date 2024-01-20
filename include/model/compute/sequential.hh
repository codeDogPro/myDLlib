#pragma once

#include <data/tensor.hh>


namespace dl{

template<typename Tp=f32>
class Sequential : public Function<Tp> {
public:
  template<typename... Args>
  Sequential(Args... args){
    _add_fn(args...);
  }

  ~Sequential(){
    for(auto &func : functions){
      delete func;
    }
  }
  
  virtual std::shared_ptr<Tensor<Tp>> 
  forward(const std::shared_ptr<Tensor<Tp>> input){
    std::shared_ptr<Tensor<Tp>> _input = input;
    std::shared_ptr<Tensor<Tp>> output;
    for(auto &func : functions){
      output = func->forward(_input);
      // std::cout << "output:\n" << *output << std::endl;
      _input = output;
      // std::cout << "input:\n" << *output << std::endl;
    }
    return output;
  }

private:
  void _add_fn(){ }

  template<typename... Args>
  void _add_fn(Function<Tp> *fn, Args... args){
    functions.push_back(fn);
    _add_fn(args...);
  }

  std::vector<Function<Tp> *> functions;
};

}