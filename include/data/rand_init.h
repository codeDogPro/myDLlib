#pragma once

#include <basic/type_tool.h>

#include <random>
#include <ctime>
#include <iostream>

namespace dl{

  template<typename T> class Tensor;

  template<typename T>
  void rand_init(Tensor<T> &t){
    std::default_random_engine engine(time(0));
    std::string name = type_name<T>();
    if(name.compare("int") == 0){
      std::uniform_int_distribution<int> random(0, 1 << 15);
      for(auto &x : t.get_data()){ x = random(engine); }
    }
    if(name.compare("float") == 0 || name.compare("double") == 0){
      std::uniform_real_distribution<float> random(0, 1);
      for(auto &x : t.get_data()){ x = random(engine); }
    }
  }

}
