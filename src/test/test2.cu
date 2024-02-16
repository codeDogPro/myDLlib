#include <dl.cuh>

using namespace dl;


template<typename T>
void resnet50_test(){
  auto input = std::make_shared<Tensor<T>>(224, 224, 3, 1);
  std::cout << "input:\n" << *input;
  const int size = 4;
  auto conv7x7 = new Conv2D<T>(7, 3, 16 * size, 2, 3, Device::CPU);
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

  auto globalAvgPool = new globalAvgPool2D<T>();
  auto flatten = new Flatten<T>();
  auto fc = new Linear<T>(2048, 10);
  auto softmax = new Softmax<T>(0);
  
  Sequential<T> resnet50(
    conv7x7, relu1,
    group1_1, group1_2, group1_3, maxPool_1, relu2,
    group2_1, group2_2, group2_3, group2_4, maxPool_2, relu3,
    group3_1, group3_2, group3_3, group3_4, group3_5, group3_6, maxPool_3, relu4,
    group4_1, group4_2, group4_3, maxPool_4, relu5,
    globalAvgPool, flatten, fc, softmax
  );

  {
    // Timer timer_compute;
    auto output = resnet50(input);
    std::cout << "output:\n" << *output << std::endl;
  }
}

void print_test(){
  auto input = std::make_shared<Tensor<f32>>(9, 20, 2, 2);
  // input->to(Device::CUDA);
  // input->to(Device::CPU);
  std::cout << "input:\n" << *input;
}

void calculator_benchmark(int n){
  Tensor<f32> a(320, 302, 80, 8, Device::CPU, 3.1);  
  Tensor<f32> b(1, 302, 80, 8, Device::CPU, 1.2);
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

  a.to(Device::CPU);
  b.to(Device::CPU);
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
  Tensor<f32> a(320, 302, 8, 4, Device::CPU, 3.1);  
  Tensor<f32> b(1, 302, 8, 4, Device::CPU, 1.2);
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
  auto input = std::make_shared<Tensor<T>>(8, 20, 4, 2);
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

template<typename T>
void softmax_cuda_test(){
  auto input = std::make_shared<Tensor<T>>(64, 4 * 64, 3, 2);
  std::cout << "input:\n" << *input;
  input->to(Device::CUDA);

  Softmax<T> softmax(0);  // COL
  auto output1 = softmax.forward(input);
  // output1->setFullPrintMode(true);
  std::cout << "output1:\n" << *output1;
  auto sum = output1->sum(0, true);
  sum->setFullPrintMode(true);
  std::cout << "sum:\n" << *sum;
}

template<typename T>
void softmax_benchmark(int n){
  auto input = std::make_shared<Tensor<T>>(2 * 79, 2 * 67, 49, 18);
  Softmax<T> softmax(0);  // COL
  // cpu
  {
    Timer t;
    for(int i = 0; i < n; i++){
      auto output = softmax(input);
    }
  }

  // gpu
  input->to(Device::CUDA);
  {
    Timer t;
    for(int i = 0; i < n; i++){
      auto output = softmax(input);
    }
  }
}

template<typename T>
void operator_cuda_test(){
  auto input = std::make_shared<Tensor<T>>(64, 4 * 64, 3, 2, Device::CPU, 2.1f);
  std::cout << "input:\n" << *input;
  input->to(Device::CUDA);
  auto sum0 = input->sum(0, true);
  auto mean0 = input->mean(0, true);
  std::cout << "sum:\n" << *sum0;
  std::cout << "mean:\n" << *mean0;
}

void init_test(){
  auto input = std::make_shared<Tensor<f32>>(4, 4, 3, 2);
  std::cout << *input;
  auto input_cuda 
    = std::make_shared<Tensor<f32>>(4, 4, 3, 2, Device::CUDA);
  std::cout << *input_cuda;
}

template<typename T>
void globalAvgPool2D_test(){
  // auto input = std::make_shared<Tensor<T>>(4*67, 4*67, 3, 2, Device::CUDA);
  auto input = std::make_shared<Tensor<T>>(4*67, 4*67, 3, 2, Device::CPU);
  // std::cout << *input;
  // input->to(Device::CUDA);
  globalAvgPool2D<T> gavgpool;
  auto output = gavgpool(input);
  std::cout << "output:\n" << *output;
  auto mean = input->mean(0)->mean(0);
  std::cout << "mean:\n" << *mean;
}

int main(){
  // print_test();                  // pass
  resnet50_test<f32>();          // pass
  // calculator_benchmark(100);     // gpu 2.1x faster than cpu(with parallel and simd)
  // calculator_test();             // pass
  // activation_test<f32>();        // pass 
  // softmax_cuda_test<f32>();      
  // softmax_benchmark<f32>(27);    // gpu 2.0x faster than cpu (why so slow?)
  // operator_cuda_test<f32>();     // pass 2/12
  // init_test();                   // pass 
  // globalAvgPool2D_test<f32>();   // pass
}