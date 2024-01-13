#include <dl.hh>
#include <mylib.h>

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
  Tensor<int> c(3, 5, 8, -1);
  std::cout << c;
  
  std::cout << c.sum(0) << c.sum(1) << c.sum(2);
  std::cout << c.mean(0) << c.mean(1) << c.mean(2);
  std::cout << c.max(0) << c.max(1) << c.max(2);
  std::cout << c.min(0) << c.min(1) << c.min(2);
}

// += -= *= /= test
void plusequal_test(){
  Tensor<int> c(3, 2, 13, 3);
  Tensor<int> d(3, 2, 13, 7);
  c += d; std::cout << c;
  c -= d; std::cout << c;
  c *= d; std::cout << c;
  c /= d; std::cout << c;
}

// +-*/ test
void plus_test(){
  Tensor<float> a(16, 32, 4, 3, 3.1);  
  Tensor<float> b(1, 32, 4, 3, 1.2);
  // Tensor<int> a(4, 4, 8, 3);
  // Tensor<int> b(4, 4, 8, 1);
  std::cout << a << b;
  auto c = a + b;
  auto d = a - b; 
  auto e = a * b; 
  auto f = a / b; 
  // std::cout << *c << *d << *e << *f;
  // std::cout << *f;
  // f->shape();
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

void reshape_test(){
  std::vector shape{3, 2, 4};
  Tensor<float> a(2, 3, 5, 4);
  std::cout << a;
  // a.reshape(3, 2, 4); a.reshape(shape); std::cout << a; }

}


void nn_test(){
  // model.push_back(new Relu<float>(true));
  // model.push_back(new Sigmoid<float>(true));
  auto input = std::make_shared<Tensor<f32>>(32, 32, 3, 1);
  // std::cout << "input:\n" << *input;
  Conv2D<float> conv2d(3, 3, 2, 1, 1);
  auto output = conv2d.forward(input);
  std::cout << "output:\n" << *output << std::endl;
  // std::cout << output.use_count() << std::endl;
  // auto output = model[0]->forward(input);
  // auto output_act = model[1]->forward(output);
  // std::cout << output << output_act;
}

int main(){
  // plus_test();
  // plusequal_test();
  // smmm_test();
  // reshape_test();
  nn_test();
  return 0;
}