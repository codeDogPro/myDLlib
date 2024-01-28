#pragma once

#include <basic/type_tool.cuh>

#include <random>
#include <ctime>
#include <iostream>

namespace dl{
  // #define DEBUG_INIT

  template<typename T> class Tensor;

  template<typename T>
  void rand_init(Tensor<T> &t){
    static std::default_random_engine engine(time(0));
    std::string name = type_name<T>();
    if(name.compare("int") == 0){
      std::uniform_int_distribution<int> random(-1 << 3, 1 << 3);
      for(auto &x : t.get_data()){ x = random(engine); 
      #ifdef DEBUG_INIT
        std::cout << x << ' ';
      #endif
      }
    }
    if(name.compare("float") == 0 || name.compare("double") == 0){
      std::uniform_real_distribution<float> random(-1.5f, 1.5f);
      for(auto &x : t.get_data()){ 
        x = random(engine);
        x = x > 0 ? x + 0.1f : x - 0.1f;
      #ifdef DEBUG_INIT
        std::cout << x << ' ';
      #endif
      }
    }
  #ifdef DEBUG_INIT
    puts("");
  #endif
  }

}
