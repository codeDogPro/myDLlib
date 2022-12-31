#include <dl.hpp>

using namespace dl;

// why all randinit tensor's data are totally same????

int main(){
  // std::vector shape{3, 3};
  // std::vector data{1, 1, 1, 4, 5, 6, 7, 8, 9};
  // Tensor<int> a(data, shape);

  // std::vector data_b{1, 3, 7, 2, 1, 3, 4, 11, 5};
  // Tensor<int> b(data_b, shape);

  // a.shape();

  // Tensor<int> a(8,4);
  // Tensor<int> c(30, 40, 512 * 8, 2);
  Tensor<int> c(4, 2,  5, 0);
  // Tensor<int> d(20, 40, 512, 4);

  // c.sum(1);
  std::cout << c;
  std::cout << c.sum(0) << c.sum(1) << c.sum(2);
  std::cout << c.mean(0) << c.mean(1) << c.mean(2);
  std::cout << c.max(0) << c.max(1) << c.max(2);
  std::cout << c.min(0) << c.min(1) << c.min(2);
  // std::cout << c.sum(0, false);
  // c += d; //std::cout << c;
  // c -= d; //std::cout << c;
  // c *= d; //std::cout << c;
  // c /= d; //std::cout << c;
  // c %= d; //std::cout << c;

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
  
  // std::vector<int> a;
  
  // std::vector<Function<int> *> model;
  // model.push_back(new Linear<int>(3, 2));
  // Tensor<int> input(1, 3, 1, 2);
  // std::cout << input;
  // auto output = model[0]->forward(input);
  // std::cout << output;
  return 0;
}