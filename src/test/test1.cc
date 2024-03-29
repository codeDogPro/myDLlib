#include <dl.hh>

using namespace dl;

// sum(), mean(), max(), min(). test
void smmm_test(){
  Tensor<int> c(3, 5, 8, 1);
  std::cout << c;
  auto sum0 = c.sum(0, true);
  auto mean0 = c.mean(0, true); 
  auto max0 = c.max(0, true); 
  auto min0 = c.min(0, true); 
  std::cout << *sum0;
  std::cout << *mean0;
  std::cout << *max0;
  std::cout << *min0;
  auto sum1 = c.sum(1, true);
  auto mean1 = c.mean(1, true); 
  auto max1 = c.max(1, true); 
  auto min1 = c.min(1, true); 
  std::cout << *sum1;
  std::cout << *mean1;
  std::cout << *max1;
  std::cout << *min1;
  auto sum2 = c.sum(2, true);
  auto mean2 = c.mean(2, true); 
  auto max2 = c.max(2, true); 
  auto min2 = c.min(2, true); 
  std::cout << *sum2;
  std::cout << *mean2;
  std::cout << *max2;
  std::cout << *min2;
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
  Tensor<float> a(68, 32, 43, 3, 3.1);  
  Tensor<float> b(1, 32, 43, 3, 1.2);
  std::cout << a << b;
  auto c = a + b;
  auto d = a - b; 
  auto e = a * b; 
  auto f = a / b; 
  std::cout << *c << *d << *e << *f;
}
 
void reshape_test(){
  std::vector shape{3, 2, 4};
  Tensor<float> a(2, 3, 5, 4);
  std::cout << a;
  // a.reshape(3, 2, 4); a.reshape(shape); std::cout << a; }

}

void conv_test(){
  auto input = std::make_shared<Tensor<int>>(8, 8, 3, 3);
  std::cout << "input:\n" << *input;
  Conv2D<int> conv2d(3, 3, 8, 1, 1);
  auto output = conv2d.forward(input);
  std::cout << "output:\n" << *output << std::endl;
}

void maxpool_test(){
  auto input = std::make_shared<Tensor<f32>>(7, 7, 3, 2);
  std::cout << "input:\n" << *input;
  MaxPool2D maxpool2d(2, 0);
  AvgPool2D avgpool2d(2, 0);
  auto output1 = maxpool2d.forward(input);
  auto output2 = avgpool2d.forward(input);
  std::cout << "output1:\n" << *output1 << std::endl;
  std::cout << "output2:\n" << *output2 << std::endl;
}


void softmax_test(){
  auto input = std::make_shared<Tensor<f32>>(4, 4, 3, 1);
  std::cout << "input:\n" << *input;
  Softmax sm0(0);
  auto output0 = sm0.forward(input);
  std::cout << "output0:\n" << *output0 << std::endl;
  Softmax sm1(1);
  auto output1 = sm1.forward(input);
  std::cout << "output1:\n" << *output1 << std::endl;
  Softmax sm2(2);
  auto output2 = sm2.forward(input);
  std::cout << "output2:\n" << *output2 << std::endl;
}

void flatten_test(){
  auto input = std::make_shared<Tensor<f32>>(4, 4, 3, 1);
  std::cout << "input:\n" << *input;
  Flatten ft(1, 3);
  auto output = ft.forward(input);
  std::cout << "output:\n" << *output;
}

void linear_test(){
  auto input = std::make_shared<Tensor<f32>>(4, 4, 3, 1);
  std::cout << "input:\n" << *input;
  Flatten ft(1, 3);
  auto output0 = ft.forward(input);
  std::cout << "output0:\n" << *output0;
  Linear fc(4 * 4 * 3, 10);
  auto output1 = fc.forward(output0);
  std::cout << "output1:\n" << *output1;
}

void residual_test(){
  auto input = std::make_shared<Tensor<f32>>(4, 4, 32, 1);
  std::cout << "input:\n" << *input;
  ResidualBlock_bottle res_block(32, 32, 64);
  auto output = res_block.forward(input);
  std::cout << "output:\n" << *output;
}

void matMul_test(){
  auto a = std::make_shared<Tensor<f32>>(16, 4, 1, 1, 0.2);
  auto b = std::make_shared<Tensor<f32>>(4, 8, 1, 1, 0.3);
  std::cout << "a:\n" << *a;
  std::cout << "b:\n" << *b;
  auto output = matMul<f32>(a, b);
  std::cout << "output:\n" << *output;
}

void matTranspose_test(){
  auto input = std::make_shared<Tensor<f32>>(8, 8, 4, 1);
  std::cout << "input:\n" << *input;
  auto output = matTranspose(input);
  // output->setFullPrintMode(true);
  std::cout << "output:\n" << *output;
}

void copy_test(){
  auto a = std::make_shared<Tensor<f32>>(5, 19, 2, 1, 0.2);
  auto b = std::make_shared<Tensor<f32>>(5, 19, 2, 1, 0.3);
  std::cout << "a:\n" << *a;
  std::cout << "b:\n" << *b;
  *b = *a;
  std::cout << "b:\n" << *b;
}

// #include <opencv2/opencv.hpp>
// void cvMat2Tensor_test(){
//   cv::Mat img = cv::imread("imgs/img3.png");
//   auto tensor = to_Tensor<f32>(img);
//   std::cout << "tensor:\n" << *tensor;
//   std::cout << img.size << std::endl;
//   std::cout << img.channels() << std::endl;
//   Conv2D conv_pad(3, 3, 8, 1, 1);
//   MaxPool2D pool1(2);
//   auto output1 = conv_pad.forward(tensor);
//   auto output2 = pool1.forward(output1);
//   std::cout << "output1:\n" << *output2 << std::endl;
//   tensor->shape();
//   output1->shape();
//   output2->shape();
// }

void print_test(){
  auto input = std::make_shared<Tensor<f32>>(120, 19, 80, 2);
  std::cout << "input:\n" << *input;
}

void sequential_test(){
  auto input = std::make_shared<Tensor<f32>>(8, 8, 3, 1);
  std::cout << "input:\n" << *input;
  auto conv2d_1 = new Conv2D(3, 3, 8, 2, 1);
  auto maxpool = new MaxPool2D(2);
  auto relu = new Relu();
  auto conv2d_2 = new Conv2D(3, 8, 16, 2, 1);
  Sequential seq(conv2d_1, maxpool, relu, conv2d_2, maxpool, relu);
  auto output = seq.forward(input);
  std::cout << "output:\n" << *output << std::endl;
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
  plus_test();         // pass
  // plusequal_test();    // pass
  // smmm_test();         // pass
  // reshape_test();      // pass
  // conv_test();         // pass
  // maxpool_test();      // pass
  // softmax_test();      // pass
  // linear_test();       // pass
  // residual_test();     // pass
  // matMul_test();       // pass
  // matTranspose_test(); // pass
  // copy_test();         // pass
  // cvMat2Tensor_test(); // pass
  // print_test();        // pass
  // sequential_test();   // pass
  // resnet50_test();     // pass
  return 0;
}