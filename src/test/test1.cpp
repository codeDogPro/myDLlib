#include <data/tensor.h>
#include <iostream>
#include <vector>

using namespace dl;

// why all randinit tensor's data are totally same????

int main(){
  // std::vector shape{3, 3};
  // std::vector data{1, 1, 1, 4, 5, 6, 7, 8, 9};
  // Tensor<int> a(data, shape);

  // std::vector data_b{1, 3, 7, 2, 1, 3, 4, 11, 5};
  // Tensor<int> b(data_b, shape);

  // a.shape();

  // // Tensor<int> a(8,4);
  Tensor<int> c(2, 4, 1, 2);
  Tensor<int> d(2, 4, 1, 3);

  std::cout << c;
  c += d; std::cout << c;
  c -= d; std::cout << c;
  c *= d; std::cout << c;
  // c /= d; std::cout << c;
  c %= d; std::cout << c;

  // std::cout << c << c + 1 << c - 3 << c * 4 << c / 3;

  // Tensor<int> c(8, 8);

  // std::cout << a << b;

  // auto c = a + b;
  // auto d = a - b; 
  // auto e = a * b; 
  // auto f = a / b; 

  // std::cout << c << d << e << f;

  // Tensor<float> c(4,4);
  // Tensor<float> d(4,4);
  // std::cout << c << d << c + d;

  // Tensor<int> a(10000, 10000, 1, 3);

  // auto b = a + 4;

  return 0;
}