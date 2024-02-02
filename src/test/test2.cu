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

void resnet50_test(){
  auto input = std::make_shared<Tensor<f32>>(224, 224, 3, 1);
  std::cout << "input:\n" << *input;
  int size = 4;
  auto conv7x7 = new Conv2D(7, 3, 16 * size, 2, 3);
  auto maxPool_1 = new MaxPool2D(3, 1, 2), maxPool_2 = new MaxPool2D(3, 1, 2);
  auto maxPool_3 = new MaxPool2D(3, 1, 2), maxPool_4 = new MaxPool2D(3, 1, 2);
  auto relu1 = new Relu(), relu2 = new Relu();
  auto relu3 = new Relu(), relu4 = new Relu();
  auto relu5 = new Relu();

  auto group1_1 = new Sequential(
    new ResidualBlock_bottle(16 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size)
  );
  auto group1_2 = new Sequential(
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size)
  );
  auto group1_3 = new Sequential(
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size),
    new ResidualBlock_bottle(64 * size, 16 * size, 64 * size)
  );

  auto group2_1 = new Sequential(
    new ResidualBlock_bottle(64 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size)
  );
  auto group2_2 = new Sequential(
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size)
  );
  auto group2_3 = new Sequential(
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size)
  );
  auto group2_4 = new Sequential(
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size),
    new ResidualBlock_bottle(128 * size, 32 * size, 128 * size)
  );                                       
                                           
  auto group3_1 = new Sequential(          
    new ResidualBlock_bottle(128 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_2 = new Sequential(          
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_3 = new Sequential(          
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_4 = new Sequential(          
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_5 = new Sequential(          
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size)
  );                                       
  auto group3_6 = new Sequential(          
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size),
    new ResidualBlock_bottle(256 * size, 64 * size, 256 * size)
  );

  auto group4_1 = new Sequential(
    new ResidualBlock_bottle(256 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size)
  );                                  
  auto group4_2 = new Sequential(     
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size)
  );                                  
  auto group4_3 = new Sequential(     
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size),
    new ResidualBlock_bottle(512 * size, 128 * size, 512 * size)
  );
  
  Sequential resnet50(
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

int main(){
  // print_test();
  plus_test();
  // resnet50_test();
}