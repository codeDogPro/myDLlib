#ifndef RAND_INIT_H
#define RAND_INIT_H

namespace dl{

  template<typename T> class Tensor;

  void rand_init(Tensor<int> & t);
  void rand_init(Tensor<float> & t);
  void rand_init(Tensor<double> & t);

}

#endif
