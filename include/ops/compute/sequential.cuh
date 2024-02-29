#pragma once

#include <basic/function.cuh>


namespace dl{

template<typename Tp=f32>
class Sequential : public Function<Tp> {
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
  
  #define SEQUENTIAL_DEBUG
  virtual std::shared_ptr<Tensor<Tp>> 
  forward(const std::shared_ptr<Tensor<const Tp>> input){
    std::shared_ptr<Tensor<Tp>> _input = input;
    std::shared_ptr<Tensor<Tp>> output;
    for(auto &func : functions){
      output = func->forward(_input);
      /*
       * After using the layer, just free the layer weight in memory_clean mode
       */
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

  std::shared_ptr<Tensor<Tp>> 
  operator()(const std::shared_ptr<Tensor<const Tp>> input){
    return forward(input);
  }

private:
  void _add_fn(){ }

  template<typename... Args>
  void _add_fn(Function<Tp> *fn, Args... args){
    functions.push_back(fn);
    _add_fn(args...);
  }

  bool memory_clean;
  std::vector<Function<Tp> *> functions;
};

}