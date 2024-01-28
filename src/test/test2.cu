#include <dl.cuh>

using namespace dl;

void print_test(){
  auto input = std::make_shared<Tensor<f32>>(120, 19, 80, 2);
  std::cout << "input:\n" << *input;
}

int main(){
  print_test();
}