#pragma once

#include <basic/function.cuh>


namespace dl{

template<typename T=f32>
class Sequential : public Function<T> {
public:
  template<typename... Args>
  Sequential(Args... args){
    _add_fn(args...);
    memory_clean = false;
  }

  virtual ~Sequential(){
    if(memory_clean == false){
      for(auto &func : functions){
        delete func;
      }
    }
  }

  bool setMemClean(bool mode){
    memory_clean = mode; 
    return mode;
  }
  
  // #define SEQUENTIAL_DEBUG
  virtual std::shared_ptr<Tensor<T>> 
  forward(const std::shared_ptr<const Tensor<T>> input){
    std::shared_ptr<const Tensor<T>> _input = input;
    std::shared_ptr<Tensor<T>> output;
    for(auto &func : functions){
      output = func->forward(_input);
      //* After using the layer, just free the layer weight in memory_clean mode
      if(memory_clean == true){
        delete func;
      }
    #ifdef SEQUENTIAL_DEBUG
      output->shape();
      // std::cout << "output:\n" << *output << std::endl;
      // std::cout << "input:\n" << *output << std::endl;
    #endif
      _input = output;
    }
    return output;
  }

  std::shared_ptr<Tensor<T>> 
  operator()(const std::shared_ptr<const Tensor<T>> input){
    return forward(input);
  }

private:
  void _add_fn(){ }

  template<typename... Args>
  void _add_fn(Function<T> *fn, Args... args){
    functions.push_back(fn);
    _add_fn(args...);
  }

  bool memory_clean;
  std::vector<Function<T> *> functions;
};

}