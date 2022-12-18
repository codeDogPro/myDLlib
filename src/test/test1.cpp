#include <data/tensor.h>
#include <iostream>
#include <vector>

using namespace dl;


int main(){
  std::vector<int> shape({2, 2});
  Tensor<int> a(shape);
  Tensor<int> b(1, 2);

  auto c = a + b; 

}