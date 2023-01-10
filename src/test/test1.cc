#include <dl.hh>

using namespace dl;

// Initialization test
void initialization_test(){
  std::vector shape{3, 3};
  std::vector data{1, 1, 1, 4, 5, 6, 7, 8, 9};
  Tensor<int> a(data, shape);
  std::vector data_b{1, 3, 7, 2, 1, 3, 4, 11, 5};
  Tensor<int> b(data_b, shape);
}

// sum(), mean(), max(), min(). test
void smmm_test(){
  // Tensor<int> c(3, 2, 2, 3);
  Tensor<int> c(3, 5, 8, 2, -1);
  std::cout << c;
  
  std::cout << c.sum(0) << c.sum(1) << c.sum(2);
  std::cout << c.mean(0) << c.mean(1) << c.mean(2);
  std::cout << c.max(0) << c.max(1) << c.max(2);
  std::cout << c.min(0) << c.min(1) << c.min(2);
}

// += -= *= /= test
void plusequal_test(){
  Tensor<int> c(3, 2, 2, 3);
  Tensor<int> d(3, 2, 2, 7);
  c += d; std::cout << c;
  c -= d; std::cout << c;
  c *= d; std::cout << c;
  c /= d; std::cout << c;
  c %= d; std::cout << c;
}

// +-*/ test
void plus_test(){
  Tensor<int> a(4, 4, 8, 2, 3);
  Tensor<int> b(4, 4, 8, 2, 1);
  std::cout << a << b;
  auto c = a + b;
  auto d = a - b; 
  auto e = a * b; 
  auto f = a / b; 
  std::cout << c << d << e;
}
 
// copy test
void copy_test(){
  Tensor<int> a(2,2,1);
  Tensor<int> b(a);
  b.shape();
  std::cout << a << b;
  a[1] = 2, b[1] = 4;
  std::cout << a << b;
}

// nn function test
void nn_test(){
  std::vector<Function<int> *> model;
  model.push_back(new Conv2D<int>(5, 2, 2, 1));
  // Tensor<int> input(4, 3, 2, 1);
  Tensor<int> input(7, 7, 2, 4);
  std::cout << input;
  auto output = model[0]->forward(input);
  // std::cout << output;
}

  
int main(){
  plus_test();
  // nn_test();
  // smmm_test();
  return 0;
}