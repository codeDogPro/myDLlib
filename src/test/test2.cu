#include "basic/tensor_macro.cuh"
#include <dl.cuh>

using namespace dl;


template<typename T>
void resnet50_test(){
  // * input:224x224x3
  // *  bs=1:  CUDA Infer: 3.20259s CPU Infer: 7.72452s
  // *  bs=6:  CUDA Infer: 19.0440s CPU Infer: 50.2053s
  // *  bs=32: CUDA Infer: 118.356s CPU Infer: 265.865s
  Device device = Device::CUDA;
  auto input = std::make_shared<Tensor<T>>(224, 224, 3, 1, device);
  // std::cout << "input:\n" << *input;
  const int size = 4;
  auto conv7x7 = new Conv2D<T>(7, 3, 16 * size, 2, 3, device);
  auto maxPool_1 = new MaxPool2D<T>(3, 1, 2), maxPool_2 = new MaxPool2D<T>(3, 1, 2);
  auto maxPool_3 = new MaxPool2D<T>(3, 1, 2), maxPool_4 = new MaxPool2D<T>(3, 1, 2);
  auto relu1 = new Relu<T>(), relu2 = new Relu<T>();
  auto relu3 = new Relu<T>(), relu4 = new Relu<T>();
  auto relu5 = new Relu<T>();

  auto group1_1 = new Sequential<T>(
    new ResidualBlock_bottle<T>(16 * size, 16 * size, 64 * size, device),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device)
  );
  auto group1_2 = new Sequential<T>(
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device)
  );
  auto group1_3 = new Sequential<T>(
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device),
    new ResidualBlock_bottle<T>(64 * size, 16 * size, 64 * size, device)
  );

  auto group2_1 = new Sequential<T>(
    new ResidualBlock_bottle<T>(64 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device)
  );
  auto group2_2 = new Sequential<T>(
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device)
  );
  auto group2_3 = new Sequential<T>(
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device)
  );
  auto group2_4 = new Sequential<T>(
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device),
    new ResidualBlock_bottle<T>(128 * size, 32 * size, 128 * size, device)
  );                                       
                                           
  auto group3_1 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(128 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device)
  );                                       
  auto group3_2 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device)
  );                                       
  auto group3_3 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device)
  );                                       
  auto group3_4 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device)
  );                                       
  auto group3_5 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device)
  );                                       
  auto group3_6 = new Sequential<T>(          
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device),
    new ResidualBlock_bottle<T>(256 * size, 64 * size, 256 * size, device)
  );

  auto group4_1 = new Sequential<T>(
    new ResidualBlock_bottle<T>(256 * size, 128 * size, 512 * size, device),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device)
  );                                  
  auto group4_2 = new Sequential<T>(     
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device)
  );                                  
  auto group4_3 = new Sequential<T>(     
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device),
    new ResidualBlock_bottle<T>(512 * size, 128 * size, 512 * size, device)
  );

  auto globalAvgPool = new globalAvgPool2D<T>();
  auto flatten = new Flatten<T>();
  auto fc = new Linear<T>(2048, 10, device);
  auto softmax = new Softmax<T>(0);
  
  Sequential<T> resnet50(
    conv7x7, relu1,
    group1_1, group1_2, group1_3, maxPool_1, relu2,
    group2_1, group2_2, group2_3, group2_4, maxPool_2, relu3,
    group3_1, group3_2, group3_3, group3_4, group3_5, group3_6, maxPool_3, relu4,
    group4_1, group4_2, group4_3, maxPool_4, relu5,
    globalAvgPool, flatten, fc, softmax
  );
  resnet50.setMemClean(true);

  TICK(Infer);
  auto output = resnet50(input);
  TOCK(Infer);
  // std::cout << "output:\n" << *output << std::endl;
}

void print_test(){
  // pass n and ch > 1. 
  // pass n == 1, ch > 1 and row > 1
  // pass n and ch == 1, row > 1
  // pass n and ch, row == 1
  // pass n > 1, ch and row == 1, col > 1
  // pass n > 1, ch > 1, row == 1, col > 1
  auto input = std::make_shared<Tensor<f32>>(1, 5, 1, 20);
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

template<typename T>
void conv_cuda_test(){
  //* pass 6x6     inum=1, ich=1, och=2, ksz=3, stride=1  randomly
  //* pass 6x6     inum=1, ich=2, och=4, ksz=3, stride=1  randomly
  //* pass 6x6     inum=2, ich=2, och=4, ksz=3, stride=1  randomly
  //* pass 6x6     inum=2, ich=1, och=4, ksz=3, stride=1  randomly
  //* pass 6x6     inum=2, ich=1, och=4, ksz=3, stride=2  randomly
  //* pass 6x6     inum=2, ich=2, och=4, ksz=3, stride=2  randomly
  //* pass 32x32   inum=1, ich=1, och=2, ksz=3, stride=2  randomly
  //  32x32   inum=2, ich=2, och=4, ksz=3, stride=2 
  //* pass 224x224 inum=2, ich=2, och=4, ksz=3, stride=2   but cpu version has bug
  //  224x224 inum=1, ich=1, och=2, ksz=3, stride=2   线程块边界有问题
  Device cuda = Device::CUDA;
  Device cpu = Device::CPU;
  Conv2D<T> conv(3, 2, 4, 1, 0, cpu);
  auto input = std::make_shared<Tensor<T>>(6, 6, 2, 2, cpu);
  std::cout << "input:\n" << *input;
  auto output_cpu = conv(input);
  
  conv.to(cuda);
  input->to(Device::CUDA);
  auto output_cuda = conv(input);
  std::cout << "cpu:\n" << *output_cpu;
  std::cout << "cuda:\n" << *output_cuda;

  if((*output_cuda) == (*output_cpu)){
    printf("right\n");
  } else{
    printf("wrong!\n");
  }
}

template<typename T>
void conv_benchmark(int n){
  /*
  *1-data: input:1x512x224x224  
  *  conv: 3x3, in=512, out=1024, stride=1
  *  time took:(68322.3 ms cpu)  (15891.3 ms cuda)
  *2-data: input:1x256x224x224  
  *  conv: 3x3, in=256, out=256, stride=1
  *  all  time took:(34105.4 ms cpu)  (2884.23 ms cuda)
  *  conv time took:( 7152.4 ms cpu)  (1845.23 ms cuda)
  */

  // {
  //   Timer t;
  //   auto input = std::make_shared<Tensor<T>>(224, 224, 256, 1, Device::CPU);
  //   Conv2D<T> conv(3, 512, 1024, 1, 0, Device::CPU);
  //   auto output = conv(input);
  // }
  // auto input = std::make_shared<Tensor<T>>(224, 224, 256, 1, Device::CUDA);
  // Conv2D<T> conv(3, 256, 256, 1, 0, Device::CUDA);
  auto input = std::make_shared<Tensor<T>>(224, 224, 256, 1, Device::CPU);
  Conv2D<T> conv(3, 256, 256, 1, 0, Device::CPU);
  {
    Timer t;
    for(int i = 0; i < n; i++){
      auto output = conv(input);
    }
  }
}

template<typename T>
void linear_test(){
  Device cuda = Device::CUDA;
  auto input = std::make_shared<Tensor<T>>(1, 2048, 1, 2, cuda);
  Linear<T> fc(2048, 1024, cuda);
  auto output1 = fc(input);

  Device cpu = Device::CPU;
  input->to(cpu);
  fc.to(cpu);
  auto output2 = fc(input);
  std::cout << "input:\n" << *input;
  std::cout << "cuda:\n" << *output1;
  output2->setFullPrintMode(true);
  std::cout << "cpu:\n" << *output2;
  if((*output1) == (*output2)){
    printf("right\n");
  }else{
    printf("wrong!\n");
  }
}

template<typename T>
void linear_benchmark(int n){
 /*
  * input_cuda_init: 1.02833s
  * input_cpu_init:  0.0010252s
  * fc_cuda_init:    0.0064175s
  * fc_cpu_init:     0.207363s
  * fc_cuda_infer:   4.60257s   WTF!
  * fc_cpu_infer:    1.05612s
  */
  Device cuda = Device::CUDA;
  Device cpu = Device::CPU;
  // CUDA
  TICK(input_cuda_init)
  auto input_cuda = std::make_shared<Tensor<T>>(1, 4096, 1, 10, cuda);
  TOCK(input_cuda_init)
  TICK(fc_cuda_init)
  Linear<T> fc_cuda(4096, 2048, cuda);
  TOCK(fc_cuda_init)
  TICK(fc_cuda_infer)
  for(int i = 0; i < n; i++){
    auto output1 = fc_cuda(input_cuda);
  }
  TOCK(fc_cuda_infer)

  // CPU
  TICK(input_cpu_init)
  auto input_cpu = std::make_shared<Tensor<T>>(1, 4096, 1, 10, cpu);
  TOCK(input_cpu_init)
  TICK(fc_cpu_init)
  Linear<T> fc_cpu(4096, 2048, cpu);
  TOCK(fc_cpu_init)
  TICK(fc_cpu_infer)
  for(int i = 0; i < n; i++){
    auto output1 = fc_cpu(input_cpu);
  }
  TOCK(fc_cpu_infer)
}

template<typename T>
void matMul_test(){
  Device device = Device::CPU;
  auto a = std::make_shared<Tensor<T>>(810, 1202, 2, 2, device);
  auto b = std::make_shared<Tensor<T>>(1202, 822, 2, 2, device);
  // std::cout << "a:\n" << *a;
  // std::cout << "b:\n" << *b;
  auto output1 = matMul<T>(a, b);
  
  a->to(Device::CUDA);
  b->to(Device::CUDA);
  auto output2 = matMul<T>(a, b);
  std::cout << "output1:\n" << *output1;
  std::cout << "output2:\n" << *output2;
  if((*output1) == (*output2)){
    printf("right\n");
  }else{
    printf("wrong!\n");
  }
}

template<typename T>
void matMul_benchmark(int n){
  /*
  *1440*2202 x 2202*1440
  *cpu:  7925.85 ms 
  *cuda: 2525 ms 
  */
  Device device = Device::CPU;
  auto a = std::make_shared<Tensor<T>>(1440, 2202, 2, 2, device);
  auto b = std::make_shared<Tensor<T>>(2202, 1440, 2, 2, device);
  {
    Timer t;
    for(int i = 0; i < n; i++){
      auto output = matMul<T>(a, b);
    }
  }
  
  a->to(Device::CUDA);
  b->to(Device::CUDA);
  {
    Timer t;
    for(int i = 0; i < n; i++){
      auto output = matMul<T>(a, b);
    }
  }
}

template<typename T>
void pooling_test(){
  Device cuda = Device::CUDA;
  Device cpu = Device::CPU;
  MaxPool2D maxpool(2);
  AvgPool2D avgpool(2);
  auto input = std::make_shared<Tensor<T>>(224, 224, 4, 2, cpu);
  // auto input = std::make_shared<Tensor<T>>(6, 6, 1, 2, cpu);
  std::cout << "input:\n" << *input;
  // auto output_cpu = maxpool(input);
  auto output_cpu = avgpool(input);
  input->to(cuda);
  // auto output_cuda = maxpool(input);
  auto output_cuda = avgpool(input);
  std::cout << "cuda:\n" << *output_cuda;
  std::cout << "cpu:\n" << *output_cpu;
  if((*output_cuda) == (*output_cpu)){
    printf("right\n");
  }else{
    printf("wrong!\n");
  }
}

template<typename T>
void residual_test(){
  Device cuda = Device::CUDA;
  Device cpu = Device::CPU;
  auto input = std::make_shared<Tensor<T>>(224, 224, 3, 2, cuda);
  ResidualBlock_bottle<T> block(3, 5, 4, cuda);
  auto output_cuda = block(input);
  std::cout << "cuda:\n" << *output_cuda;
  // std::cout << "cpu:\n" << *output_cpu;
  // if((*output_cuda) == (*output_cpu)){
  //   printf("right\n");
  // }else{
  //   printf("wrong!\n");
  // }
}

template<typename T>
void k1s1_cuda_Conv2d_test(){
  //* input:4x4x1x2     kernel:1x2 randomly PASS
  //* input:4x4x3x2     kernel:3x3 randomly PASS
  //* input:12x12x3x2   kernel:3x3 randomly PASS
  //* input:12x12x1x2   kernel:1x2 randomly PASS
  //* input:224x224x1x2 kernel:1x2 randomly PASS
  //* input:224x224x3x8 kernel:3x8 randomly PASS
  Device cuda = Device::CUDA;
  Device cpu = Device::CPU;
  auto input = std::make_shared<Tensor<T>>(224, 224, 3, 2, cpu);
  Conv2D<T> conv(1, 3, 8, 1, 0, cpu);
  std::cout << "input:\n" << *input;
  auto output_cpu = conv(input);

  conv.to(cuda);
  input->to(cuda);
  auto output_cuda = conv(input);
  std::cout << "cuda:\n" << *output_cuda;
  std::cout << "cpu:\n" << *output_cpu;
  if((*output_cuda) == (*output_cpu)){
    printf("right\n");
  }else{
    printf("wrong!\n");
  }
}

int main(){
  // print_test();                  // pass
  // calculator_benchmark(100);     // gpu 2.1x faster than cpu(with parallel and simd)
  // calculator_test();             // pass
  // activation_test<f32>();        // pass 
  // softmax_cuda_test<f32>();      
  // softmax_benchmark<f32>(27);    // gpu 2.0x faster than cpu (why so slow?)
  // operator_cuda_test<f32>();     // pass 2/12
  // init_test();                   // pass 
  // globalAvgPool2D_test<f32>();   // pass
  // conv_cuda_test<f32>();         // pass 8/10;
  // conv_benchmark<f32>(10);       // 3.87x faster than cpu
  // linear_test<f32>();            // pass 
  // linear_benchmark<f32>(100);    // cpu new: 38.694 ms old: 429.975 ms 
  // matMul_test<f32>();            // pass
  // matMul_benchmark<f32>(10);     // 3.4x faster than cpu
  // pooling_test<f32>();           // pass 
  // residual_test<f32>();          // pass
  resnet50_test<f32>();          // pass
  // k1s1_cuda_Conv2d_test<f32>();  // pass
}