#include <dl.cuh>

using namespace dl;

void print_test(){
  auto input = std::make_shared<Tensor<f32>>(120, 19, 80, 2);
  input->to(Device::CUDA);
  input->to(Device::CPU);
  std::cout << "input:\n" << *input;
}

void plus_test(){
  Tensor<float> a(68, 32, 43, 3, 3.1);  
  Tensor<float> b(1, 32, 43, 3, 1.2);
  std::cout << a << b;
  auto c = a + b;
  auto d = a - b; 
  auto e = a * b; 
  auto f = a / b; 
  std::cout << *c << *d << *e << *f;
}


int main(){
  // print_test();
  plus_test();
}