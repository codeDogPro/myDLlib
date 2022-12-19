// #include <data/rand_init.h>
#include <data/tensor.h>

#include <random>
#include <ctime>

#include <assert.h>
#include <iostream>

namespace dl{

  std::default_random_engine generator;

  void rand_init(Tensor<int> &t){
    std::uniform_int_distribution<int> distribution(0, 1 << 16);
    generator.seed(time(0));
    auto random = std::bind(distribution, generator);
    for(auto &x : t.get_data()){ x = random(); }
  }

  void rand_init(Tensor<float> &t){
    std::uniform_real_distribution<float> distribution(-1, 1);
    generator.seed(time(0));
    auto random = std::bind(distribution, generator);
    for(auto &x : t.get_data()){ x = random(); }
  }

  void rand_init(Tensor<double> &t){
    std::uniform_real_distribution<double> distribution(-1, 1);
    generator.seed(time(0));
    auto random = std::bind(distribution, generator);
    for(auto &x : t.get_data()){ x = random(); }
  }

}