#include <dl.cuh>
#include <iterator>
#include <memory>
#include <type_traits>

using namespace dl;

template<typename T>
void resnet50_test(){
  auto input = std::make_shared<Tensor<T>>(224, 224, 3, 1);
  std::cout << "input:\n" << *input;
  int size = 4;
  auto conv7x7 = new Conv2D<T>(7, 3, 16 * size, 2, 3);
  auto maxPool_1 = new MaxPool2D<T>(3, 1, 2), maxPool_2 = new MaxPool2D<T>(3, 1, 2);
  auto maxPool_3 = new MaxPool2D<T>(3, 1, 2), maxPool_4 = new MaxPool2D<T>(3, 1, 2);
  auto relu1 = new Relu<T>(), relu2 = new Relu<T>();
  auto relu3 = new Relu<T>(), relu4 = new Relu<T>();
  auto relu5 = new Relu<T>();

  auto group1_1 = new Sequential<T>(
    new ResidualBlock_bottle<T>(16 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size)
  );
  auto group1_2 = new Sequential<T>(
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size)
  );
  auto group1_3 = new Sequential<T>(
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size)
  );

  auto group2_1 = new Sequential<T>(
    new ResidualBlock_bottle<T>(64 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size)
  );
  auto group2_2 = new Sequential<T>(
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size)
  );
  auto group2_3 = new Sequential<T>(
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size)
  );
  auto group2_4 = new Sequential<T>(
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size)
  );                                       
                                           
  auto group3_1 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(128 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_2 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_3 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_4 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_5 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_6 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size)
  );

  auto group4_1 = new Sequential<T>(
    new ResidualBlock_bottle<T>(256 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size)
  );                                  
  auto group4_2 = new Sequential<T>(     
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size)
  );                                  
  auto group4_3 = new Sequential<T>(     
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size)
  );
  
  Sequential<T> resnet50(
    conv7x7, relu1,
    group1_1, group1_2, group1_3, maxPool_1, relu2,
    group2_1, group2_2, group2_3, group2_4, maxPool_2, relu3,
    group3_1, group3_2, group3_3, group3_4, group3_5, group3_6, maxPool_3, relu4,
    group4_1, group4_2, group4_3, maxPool_4, relu5
  );

  {
    Timer timer_compute;
    auto output = resnet50.forward(input);
  }
  // std::cout << "output:\n" << *output << std::endl;
}

void print_test(){
  auto input = std::make_shared<Tensor<f32>>(9, 20, 2, 2);
  // input->to(Device::CUDA);
  // input->to(Device::CPU);
  std::cout << "input:\n" << *input;
}

void calculator_benchmark(int n){
  Tensor<f32> a(320, 302, 80, 8, 3.1);  
  Tensor<f32> b(1, 302, 80, 8, 1.2);
  // std::cout << a << b;
  {
    Timer t;
    a.to(Device::CUDA);
    b.to(Device::CUDA);
    for(int i = 0; i < n; i++){
      auto c = a + b;
      auto d = a - b; 
      auto e = a * b; 
      auto f = a / b; 
    }
  }

  a.to(Device::CPU, false);
  b.to(Device::CPU, false);
  {
    Timer t;
    for(int i = 0; i < n; i++){
      auto c = a + b;
      auto d = a - b; 
      auto e = a * b; 
      auto f = a / b; 
    }
  }
  // std::cout << *c << *d << *e << *f;
}

void calculator_test(){
  Tensor<f32> a(320, 302, 8, 4, 3.1);  
  Tensor<f32> b(1, 302, 8, 4, 1.2);
  std::cout << a << b;
  a.to(Device::CUDA);
  b.to(Device::CUDA);
  auto c = a + b;
  auto d = a - b; 
  auto e = a * b; 
  auto f = a / b; 
  std::cout << *c << *d << *e << *f;
}

template <typename T>
void activation_test(){
  auto input = std::make_shared<Tensor<T>>(8, 8, 4, 2);
  std::cout << "input:\n" << *input;
  input->to(Device::CUDA);

  Relu<T> relu;
  Sigmoid<T> sigmoid;
  Softmax<T> softmax(0);

  auto output1 = relu.forward(input);
  auto output2 = sigmoid.forward(input);
  // auto output3 = softmax.forward(input);

  std::cout << "output1:\n" << *output1;
  std::cout << "output2:\n" << *output2;
  // std::cout << "output3:\n" << *output3;
}


int main(){
  // print_test();            // pass
  // resnet50_test<f32>();    // pass
  // calculator_benchmark(100); // gpu 2.1x faster than cpu(with parallel and simd)
  // calculator_test();       // pass
  // activation_test<f32>();  // pass 
}