#include <data/tensor.h>
#include <iostream>
#include <vector>

using namespace dl;


int main(){
  std::vector<int> shape({3, 2});
  Tensor<int> a(shape);
  Tensor<int> b(1, 2);

  std::cout << a << b;

  auto c = a + b; 

  std::cout << c;

  return 0;
}