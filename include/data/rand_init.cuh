#pragma once

#include <basic/type_tool.cuh>
#include <basic/tensor_macro.cuh>
#include <data/align_alloc.cuh>

// cuda lib
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>

#include <random>
#include <ctime>
#include <cstdlib>

// #define DEBUG_INIT
#ifdef DEBUG_INIT
#include <iostream>
#endif

namespace dl{
  template<typename T> class Tensor;

  template<typename T=f32>
  void rand_init_cpu(thrust::host_vector<T, AlignedAllocator<T, 64>> &data){
    static std::default_random_engine _engine(time(0));
    std::string name = type_name<T>();
    if(name.compare("int") == 0){
      std::uniform_int_distribution<int> random(-1 << 3, 1 << 3);
      for(T &x : data){ 
        x = random(_engine); 
      }
    }
    else if(name.compare("float") == 0){
      std::normal_distribution<float> random(0.0f, 1.0f);
      for(T &x : data){ 
        x = random(_engine);
        if(x > 0 && x <  0.01) x += 0.01;
        if(x < 0 && x > -0.01) x -= 0.01;
      }
    }
  }

  template<typename T>
  __global__ void
  add_bias(thrust::device_ptr<T> data, int n){
    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = begin; i < n; i += stride){
      T x = data[i];
      if(x > 0 && x <  0.01) data[i] = x + 0.01;
      if(x < 0 && x > -0.01) data[i] = x - 0.01;
    }
  }

  bool __isFirst = true;

  template<typename T=f32>
  void rand_init_cuda(thrust::device_vector<T> &data){
    auto ptr = data.data().get();

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    if(__isFirst == true){
      srand(time(0));
      __isFirst = false;
    }
    curandSetPseudoRandomGeneratorSeed(gen, rand());

    std::string name = type_name<T>();
    if(name.compare("int") == 0){
      //TODO: have bug?
      curandGenerate(gen, (ui32 *)ptr, data.size());
    }
    else if(name.compare("float") == 0){
      curandGenerateNormal(gen, (f32 *)ptr, data.size(), 0.0f, 1.0f);
      add_bias<T><<<64, 128>>>(data.data(), data.size());
    }
    else if(name.compare("double") == 0){
      curandGenerateNormalDouble(gen, (f64 *)ptr, data.size(), 0.0f, 1.0f);
      add_bias<T><<<64, 128>>>(data.data(), data.size());
    }
  }
}
