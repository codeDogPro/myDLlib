#pragma once

#include <basic/type_tool.cuh>
#include <basic/tensor_macro.cuh>
#include <data/align_alloc.cuh>

#include <random>
#include <ctime>
#include <iostream>

// cuda lib
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace dl{
  // #define DEBUG_INIT
  template<typename T> class Tensor;

  template<typename T=f32>
  void rand_init_cpu(thrust::host_vector<T, AlignedAllocator<T, 64>> &data){
    static std::default_random_engine engine(time(0));
    std::string name = type_name<T>();
    if(name.compare("int") == 0){
      std::uniform_int_distribution<int> random(-1 << 3, 1 << 3);
      for(T &x : data){ 
        x = random(engine); 
      #ifdef DEBUG_INIT
        std::cout << x << ' ';
      #endif
      }
    }
    if(name.compare("float") == 0 || name.compare("double") == 0){
      std::uniform_real_distribution<float> random(-1.5f, 1.5f);
      for(T &x : data){ 
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

  template<typename T=f32>
  void rand_init_cuda(thrust::device_vector<T> &data){

  }
}
